import { StreamingTextResponse, LangChainStream, Message } from 'ai';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { AIMessage, HumanMessage } from 'langchain/schema';

export const runtime = 'edge';

export async function POST(req: Request) {
  const { messages } = await req.json();
  const currentMessageContent = messages[messages.length - 1].content;

  const baseUrl = process.env.BASE_URL || "http://localhost:3000";

  const vectorSearch = await fetch(`${baseUrl}/api/vectorSearch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: currentMessageContent,
  }).then((res) => res.json());

  // const TEMPLATE = `You are a very enthusiastic freeCodeCamp.org representative who loves to help people! Given the following sections from the freeCodeCamp.org contributor documentation, answer the question using only that information, outputted in markdown format. If you are unsure and the answer is not explicitly written in the documentation, say "Sorry, I don't know how to help with that."
  
  // Context sections:
  // ${JSON.stringify(vectorSearch)}

  // Question: """
  // ${currentMessageContent}
  // """
  // `;

  const TEMPLATE = `You are an expert assistant answering questions based only on the indexed content of the two academic papers related to Retrieval-Augmented Generation (RAG) and Agentic RAG.

Use retrieved context to answer user queries. Do not make up answers.

Context:
${JSON.stringify(vectorSearch)}

Question:
${currentMessageContent}

Answer:
`;

  console.log(TEMPLATE)


  messages[messages.length -1].content = TEMPLATE;

  const { stream, handlers } = LangChainStream();

  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
  });

  llm
    .call(
      (messages as Message[]).map(m =>
        m.role == 'user'
          ? new HumanMessage(m.content)
          : new AIMessage(m.content),
      ),
      {},
      [handlers],
    )
    .catch(console.error);

  return new StreamingTextResponse(stream);
}
