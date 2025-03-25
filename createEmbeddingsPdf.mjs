
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MongoClient } from "mongodb";
import path from "path";
import { promises as fsp } from "fs";
import "dotenv/config";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const dbName = "pdf_docs";
const collectionName = "embeddings";
const collection = client.db(dbName).collection(collectionName);

const docs_dir = "_assets/pdf-docs";
const fileNames = await fsp.readdir(docs_dir);
console.log("PDF files:", fileNames);

for (const fileName of fileNames) {
  const filePath = path.join(docs_dir, fileName);
  const ext = path.extname(fileName).toLowerCase();

  if (ext === ".pdf") {
    console.log(`Vectorizing PDF: ${fileName}`);
    const loader = new PDFLoader(filePath);
    const rawDocs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
      });
    const output = await splitter.splitDocuments(rawDocs);

    await MongoDBAtlasVectorSearch.fromDocuments(
      output,
      new OpenAIEmbeddings(),
      {
        collection,
        indexName: "vector_index",
        textKey: "text",
        embeddingKey: "embedding",
      }
    );
  } else {
    console.log(`Skipping non-PDF file: ${fileName}`);
  }
}

await client.close();
console.log("PDF Embedding complete.");
