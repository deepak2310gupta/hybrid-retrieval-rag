from langchain_core.prompts import ChatPromptTemplate


def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
            You are a precise question-answering assistant.

            Answer the question using ONLY the information in the context.

            Rules:
            1. If the answer is explicitly present in the context, answer it directly.
            2. Do not claim that the answer is missing when it is present.
            3. Do not add information from your own knowledge.
            4. Keep the answer concise.
            5. If the answer is not present in the context, say exactly:
            I could not find the answer in the provided documents.

            Context:
            {context}

            Question:
            {input}

            Answer:
        """
    )

    return prompt