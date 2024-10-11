import logging
import json

from const import SYSTEM_MESSAGE, EXAMPLES, REACT_MESSAGE, REACT_EXAMPLES
from lib.models import SerVerlessWorkflow
from lib.json_validator import JsonSchemaValidationException
from lib.validator import ParsedOutputException
from lib.serverless_validation import ServerlessValidation

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

class WorkflowRepo:
    def __init__(self):
        self._data = {}

    def get(self, session, default=None):
        if session in self._data and self._data[session]:
            return self._data[session][0]
        return default

    def list(self, session):
        if session in self._data and self._data[session]:
            return self._data[session]
        return []

    def put(self, session, data):
        if session not in self._data:
            self._data[session] = []
        self._data[session].insert(0, data)

    def setdefault(self, session, default=None):
        if session not in self._data or not self._data[session]:
            self.put(session, default)
            return default
        return self._data[session][0]

    def __getitem__(self, session):
        return self.get(session)

    def __setitem__(self, session, data):
        self.put(session, data)

WORKFLOWS = WorkflowRepo()

def get_prompt_details():
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=EXAMPLES,
    )

    # @TODO add chat message history here
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ]
    )
    return prompt


def format_docs(docs):
    logging.info("The number of docs added to the request are {0}".format(
        len(docs)))
    logging.debug("Files added to the prompt: {}".format(", ".join(
        doc.metadata.get('source') for doc in docs)))
    result = "\n".join("Source: {}\nContent:\n{}".format(
        doc.metadata.get('source'), doc.page_content) for doc in docs)
    return result

def get_all_workflow_for_session(ctx, session_id):
    return WORKFLOWS.list(str(session_id))

def get_workflow_for_session(ctx, session_id):
    return WORKFLOWS.get(str(session_id), {})


def get_history(ctx, session_id):
    ctx.history_repo.set_session(session_id)
    return ctx.history_repo.messages


# This function iterates over the json and try to fix it.
def set_json_response(chain, session_id, ai_response, validator):


    def set_workflow(doc, valid):
        WORKFLOWS.put(
            str(session_id),
            { "document": doc, "valid": valid})

    template = """
Please analyze and fix the following JSON workflow definition according to the provided JSON schema, JSON Schema compilation errors, and Maven compilation log:

Workflow JSON:
```json
{json_data}
```

JSON Schema Compilation Errors:
```
{validation_errors}```

Maven Compilation Log:
```
{compilation_error}
```

Please perform the following tasks:
1. Review the JSON Schema compilation errors and explain their implications.
2. Validate the JSON against the schema and list any additional validation errors not captured in the compilation errors.
3. Analyze the Maven compilation log and list only the errors.
4. Suggest fixes for all identified errors (JSON Schema compilation errors, validation errors, and Maven errors).
5. Apply the suggested fixes and provide the corrected JSON workflow definition.

Explain your reasoning for each suggested fix and any changes made to the JSON. If there are conflicting errors or fixes, prioritize them and explain your decision-making process.
"""

    prompt = PromptTemplate(
        input_variables=["json_data", "validation_errors"],
        template=template
    )
    yield "Validation started\n\n"
    for i in range(5):
        try:
            document = validator.invoke(ai_response)
            if document is None:
                logging.error("document is null")
                return
            logging.error("Finally the document is correct")
            set_workflow(document, True)
            return
        except (ParsedOutputException) as e:
            yield "<b>Parsing output stage {0} of 5, with Parsed error</b>\n\n"
            logging.error(
                f"cannot get valid JSON document from the response: {e}")
        except (JsonSchemaValidationException) as e:
            set_workflow(e.data, False)
            workflow_json = json.dumps(e.data, indent=2)
            serverless_validation, valid = ServerlessValidation(workflow_json).run()
            logging.debug("Trying to validate: {0}".format(workflow_json))
            logging.error(
                    "Workflow is not correct has some JSON issues: {0}".format(
                        e.get_error()))
            yield "<b>Parsing output stage {0} of 5, with {1} errors</b>\n\n".format(
                    i, e.get_number_of_errors())
            formatted_prompt = prompt.format(
                    json_data=workflow_json,
                    validation_errors=e.get_error(),
                    compilation_error=serverless_validation,
                    schema=SerVerlessWorkflow.schema_json())
            ai_response = chain.react(formatted_prompt)
            logging.debug(f"react response: {ai_response}")
        except Exception as e:
            yield "<b>Parsing output stage {0} of 5, with unexpected error</b>\n\n"
            logging.error(f"Latest response: {ai_response.content}")
            logging.error(f"Unexected exception on the workflow: {e}")



    yield "Validation finished"

class ChatChain():
    def get_prompt_details(cls):
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        schema_prompt = ChatPromptTemplate.from_messages([
            ("system", "Ensure that your response compiles with the following schema definition: ```{schema}```"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=EXAMPLES,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                schema_prompt,
                ("system", SYSTEM_MESSAGE),
                few_shot_prompt,
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", "{input}"),
            ]
        )

        return prompt

    def get_react_prompt(cls):

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=REACT_EXAMPLES,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed."),
                # ("system", REACT_MESSAGE),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )

        return prompt

    def __init__(self, llm, retriever, history, session_id):
        prompt = self.get_prompt_details()
        self._llm = llm
        self._history_repo = history
        self._config = {"configurable": {
            "session_id": "{0}".format(session_id)}}
        self._chain = (
            {
                "context": retriever | format_docs,
                "schema": lambda _: SerVerlessWorkflow.schema_json(),
                "input": RunnablePassthrough(),
                "history": lambda _: history.messages,
            }
            | prompt
            | llm)

    def react(self, user_message):
        prompt = self.get_react_prompt()
        # import ipdb; ipdb.set_trace()
        #return self._llm.invoke(user_message)
        chain = (
            #{
            #    #"schema": lambda _: SerVerlessWorkflow.schema_json(),
            #    "input": RunnablePassthrough(),
            #}
            { "input": RunnablePassthrough()}
            | prompt
            | self._llm)

        return chain.invoke({"input": user_message})

    @property
    def config(self):
        return self._config

    def chain(self):
        return self._chain

    def chain_with_history(self):
        return RunnableWithMessageHistory(
            self.chain(),
            lambda session_id: self._history_repo,
            input_messages_key="input",
            history_messages_key="history",
        )

    def invoke(self, data):
        return self.chain_with_history().invoke(data, config=self._config)

    def stream(self, data):
        return self.chain_with_history().stream(data, config=self._config)


def get_response_for_session(ctx, session_id, user_message):
    retriever = ctx.repo.retriever
    llm = ctx.ollama.llm
    # @TODO check if this clone the thing to a new object.
    history_repo = ctx.history_repo
    history_repo.set_session(f"{session_id}")
    yield "Waiting on AI response\n"
    chain = ChatChain(llm, retriever, history_repo, session_id)
    ai_response = []
    result = chain.stream({"input": user_message})

    for x in result:
        ai_response.append(x.content)
        yield str(x.content)

    yield "\nChecking if json is correct and validation\n\n"
    full_ai_response = "".join(ai_response)
    for x in set_json_response(chain, session_id, full_ai_response, ctx.validator):
        yield str(x)
