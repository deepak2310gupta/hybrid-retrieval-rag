from langchain_core.prompts import ChatPromptTemplate


def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
            You are a question-answering assistant.

            Use ONLY the provided context to answer the question.

            Rules:
            - If the answer is present in the context, answer directly.
            - If the answer is not present in the context, say:
            "I could not find the answer in the provided documents."

            Context:
            {context}

            Question:
            {input}

            Answer:
        """
    )

    return prompt