import { openai } from '@ai-sdk/openai';
import { streamText, tool } from 'ai';
import { z } from 'zod';
import { db } from '@/db';
import { documents } from '@/db/schema';
import { cosineDistance, desc, gt, sql } from 'drizzle-orm';
import OpenAI from 'openai';

// Initialize OpenAI client for embeddings
const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai('gpt-4'),
    messages,
    tools: {
      retrieveInformation: tool({
        description: 'Retrieve relevant financial documents and information from the knowledge base',
        parameters: z.object({
          query: z.string().describe('The search query to find relevant financial documents'),
        }),
        execute: async ({ query }) => {
          try {
            // Generate embedding for the query
            const embeddingResponse = await openaiClient.embeddings.create({
              model: 'text-embedding-ada-002',
              input: query,
            });
            
            const queryEmbedding = embeddingResponse.data[0].embedding;

            // Search for similar documents using cosine similarity
            const similarDocuments = await db
              .select({
                id: documents.id,
                title: documents.title,
                content: documents.content,
                metadata: documents.metadata,
                similarity: sql<number>`1 - (${cosineDistance(documents.embedding, queryEmbedding)})`,
              })
              .from(documents)
              .where(gt(sql`1 - (${cosineDistance(documents.embedding, queryEmbedding)})`, 0.7))
              .orderBy((t) => desc(t.similarity))
              .limit(5);

            return {
              documents: similarDocuments.map(doc => ({
                id: doc.id,
                title: doc.title,
                content: doc.content.substring(0, 1000) + (doc.content.length > 1000 ? '...' : ''),
                metadata: doc.metadata ? JSON.parse(doc.metadata) : null,
                similarity: doc.similarity,
              })),
              query,
            };
          } catch (error) {
            console.error('Error retrieving information:', error);
            return {
              documents: [],
              query,
              error: 'Failed to retrieve information from the knowledge base',
            };
          }
        },
      }),
      addDocument: tool({
        description: 'Add a new financial document to the knowledge base',
        parameters: z.object({
          title: z.string().describe('The title of the financial document'),
          content: z.string().describe('The content of the financial document'),
          metadata: z.object({}).passthrough().optional().describe('Additional metadata about the document'),
        }),
        execute: async ({ title, content, metadata }) => {
          try {
            // Generate embedding for the document content
            const embeddingResponse = await openaiClient.embeddings.create({
              model: 'text-embedding-ada-002',
              input: content,
            });
            
            const embedding = embeddingResponse.data[0].embedding;

            // Insert the document into the database
            const [newDocument] = await db
              .insert(documents)
              .values({
                title,
                content,
                embedding,
                metadata: metadata ? JSON.stringify(metadata) : null,
              })
              .returning();

            return {
              success: true,
              document: {
                id: newDocument.id,
                title: newDocument.title,
                content: newDocument.content.substring(0, 200) + '...',
              },
            };
          } catch (error) {
            console.error('Error adding document:', error);
            return {
              success: false,
              error: 'Failed to add document to the knowledge base',
            };
          }
        },
      }),
    },
  });

  return result.toDataStreamResponse();
}