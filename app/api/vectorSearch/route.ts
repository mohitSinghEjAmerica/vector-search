import { OllamaEmbeddings } from "@langchain/ollama";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import mongoClientPromise from "@/app/lib/mongodb";

export async function POST(req: Request) {
    const client = await mongoClientPromise;
    const dbName = "pdf_docs_ollama";
    const collectionName = "embeddings";
    const collection = client.db(dbName).collection(collectionName);

    const embeddings = new OllamaEmbeddings({
        baseUrl: "http://localhost:11434",
        model: "nomic-embed-text",
    });

    const question = await req.text();

    const vectorStore = new MongoDBAtlasVectorSearch(
        embeddings,
        {
            collection,
            indexName: "default",
            textKey: "text",
            embeddingKey: "embedding",
        }
    );

    const retriever = vectorStore.asRetriever();

    const retrieverOutput = await retriever._getRelevantDocuments(question);

    return new Response(JSON.stringify(retrieverOutput), {
        headers: { "Content-Type": "application/json" },
    });
}
