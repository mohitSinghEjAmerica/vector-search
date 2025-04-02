import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import { StreamingTextResponse, LangChainStream, Message } from 'ai';
// import { ChatOllama } from "langchain/chat_models/ollama";
// import { AIMessage, HumanMessage } from 'langchain/schema';

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

  const previousConversation = messages.reverse()
    .filter((m: { role: string; content: string }, idx: number) => idx < 4 && m.role === 'user')
    .reverse()

  const TEMPLATE = `You are an expert assistant answering questions based only on the indexed content of the two academic papers related to Retrieval-Augmented Generation (RAG) and Agentic RAG.

Use retrieved context to answer user queries. Do not make up answers.

Context:
${JSON.stringify(vectorSearch) + JSON.stringify(previousConversation)}

Question:
${currentMessageContent}

Answer:
`;

  messages[messages.length -1].content = TEMPLATE;

  const { stream, handlers } = LangChainStream();

  // Replace ChatOpenAI with ChatOllama
  const llm = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default Ollama server URL
    model: "gemma3:4b", // Or any other model you've pulled with Ollama
    temperature: 0.0,
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