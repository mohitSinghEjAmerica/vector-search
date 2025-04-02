import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MongoClient } from "mongodb";
import path from "path";
import { promises as fsp } from "fs";
import "dotenv/config";

// PDF metadata extractor (using pdf-lib for more robust metadata extraction)
import { readFile } from 'fs/promises';
import { PDFDocument } from 'pdf-lib';

const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const dbName = "pdf_docs_ollama";
const collectionName = "embeddings";
const collection = client.db(dbName).collection(collectionName);

const docs_dir = "_assets/pdf-docs";
const fileNames = await fsp.readdir(docs_dir);
console.log("PDF files:", fileNames);

// Initialize Ollama embeddings
const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

async function extractPdfMetadata(filePath) {
  try {
    const pdfBytes = await readFile(filePath);
    const pdfDoc = await PDFDocument.load(pdfBytes);
    
    return {
      title: pdfDoc.getTitle() || path.basename(filePath, '.pdf'),
      author: path.basename(filePath, '.pdf') === 'agentic-rag' ? "Aditi Singh, Abul Ehtesham, Saket Kumar,Tala Talaei Khoei": "Yunfan Gaoa, Yun Xiongb, Xinyu Gaob, Kangxiang Jiab, Jinliu Panb, Yuxi Bic, Yi Daia, Jiawei Suna, Meng Wangc,Haofen Wang",
      pageCount: pdfDoc.getPageCount(),
      creationDate: pdfDoc.getCreationDate()?.toString() || 'Unknown',
      modificationDate: pdfDoc.getModificationDate()?.toString() || 'Unknown'
    };
  } catch (error) {
    console.error(`Error extracting metadata from ${filePath}:`, error);
    return {
      title: path.basename(filePath, '.pdf'),
      author: 'Unknown',
      pageCount: 0,
      creationDate: 'Unknown',
      modificationDate: 'Unknown'
    };
  }
}

for (const fileName of fileNames) {
  const filePath = path.join(docs_dir, fileName);
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.pdf') {
    console.log(`Processing PDF: ${fileName}`);
    
    // Extract PDF metadata
    const pdfMetadata = await extractPdfMetadata(filePath);
    
    // Load and split documents
    const loader = new PDFLoader(filePath, {
      // Optional: if you want to preserve more PDF structure
      splitPages: true
    });
    
    const rawDocs = await loader.load();
    
    // Add metadata to each document
    const docsWithMetadata = rawDocs.map(doc => {
      return {
        ...doc,
        metadata: {
          ...doc.metadata,
          ...pdfMetadata,
          // Add PDF-specific metadata
          pdfFileName: fileName,
          // Add page info (if splitPages was true)
          pageNumber: doc.metadata.loc?.pageNumber || 1,
          // Add chunk info
          chunkIndex: doc.metadata.loc?.chunkIndex || 0
        }
      };
    });

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });
    
    const splitDocs = await splitter.splitDocuments(docsWithMetadata);

    await MongoDBAtlasVectorSearch.fromDocuments(
      splitDocs,
      embeddings,
      {
        collection,
        indexName: "vector_index",
        textKey: "text",
        embeddingKey: "embedding",
      }
    );
    
    console.log(`Processed ${fileName} with metadata:`, pdfMetadata);
  }
}

await client.close();
console.log("PDF processing complete with metadata.");