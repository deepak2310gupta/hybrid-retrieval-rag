from langchain_core.prompts import ChatPromptTemplate


def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
            You are a helpful and precise question-answering assistant.

            Your task is to answer the user's question using ONLY the information
            contained in the provided context.

            Follow these rules carefully:

            1. Ground your answer in the context.
            - Use only information explicitly supported by the context.
            - Do not use your own knowledge or make assumptions.

            2. If the answer is clearly available in the context:
            - Answer the question directly.
            - Keep the answer concise but complete.
            - Include relevant details when they help answer the question.

            3. If the context does not contain enough information to answer:
            - Respond exactly with:
                "I could not find the answer in the provided documents."

            Context:
            {context}

            Question:
            {input}

            Answer:
        """
    )

    return prompt