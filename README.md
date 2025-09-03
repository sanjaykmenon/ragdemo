# Financial Documents RAG Demo

A Next.js application that uses Retrieval-Augmented Generation (RAG) to answer questions about financial documents. Built with OpenAI's GPT-4 and embeddings, PostgreSQL for vector storage, and a modern UI.

## Features

- **AI Chat**: Ask questions about your financial documents using GPT-4
- **Document Upload**: Upload financial documents (PDF, TXT, CSV, DOCX)
- **Vector Search**: Find relevant information using embeddings
- **PostgreSQL Storage**: Store document embeddings in PostgreSQL
- **Modern UI**: Clean interface with animations
- **Streaming Responses**: Real-time AI response streaming
- **Free Hosting**: Deploy on Vercel with Neon PostgreSQL

## Quick Start

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd ragdemo
npm install
```

### 2. Set up PostgreSQL Database (Neon)

1. Create a free account at [Neon](https://neon.tech)
2. Create a new project
3. Copy your connection string: `postgresql://username:password@host/database?sslmode=require`

### 3. Get OpenAI API Key

1. Go to [OpenAI](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### 4. Configure Environment Variables

Create `.env.local`:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
DATABASE_URL=postgresql://username:password@host/database?sslmode=require

# Optional
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### 5. Set up Database

```bash
npm run db:generate
npm run db:migrate
```

### 6. Initialize Sample Data

```bash
npm run dev
```

In another terminal:

```bash
curl -X POST http://localhost:3000/api/init-docs
```

### 7. Start Using

1. Open [http://localhost:3000](http://localhost:3000)
2. Try questions like:
   - "What are the Q3 2024 financial metrics?"
   - "What are the main risk factors?"
   - "Show me cash flow information"
3. Upload your own documents

## Deployment

### Vercel

1. Push to GitHub
2. Import repository in [Vercel](https://vercel.com)
3. Add environment variables
4. Deploy

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── chat/route.ts          # Chat API with RAG
│   │   └── init-docs/route.ts     # Sample document initialization
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                   # Main chat interface
├── db/
│   ├── index.ts                   # Database connection
│   └── schema.ts                  # Database schema
```

## Tech Stack

- **Framework**: Next.js 15
- **AI**: Vercel AI SDK + OpenAI GPT-4
- **Database**: PostgreSQL (Neon) + Drizzle ORM
- **Embeddings**: OpenAI text-embedding-ada-002
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **Icons**: Lucide React

## Sample Questions

- "What was the revenue growth in Q3 2024?"
- "Analyze the cash flow trends"
- "What are the main risk factors?"
- "Show me the balance sheet highlights"

## Features

### Document Upload
- Supports PDF, TXT, CSV, DOCX
- Auto-generates embeddings
- Stores with metadata in PostgreSQL

### Semantic Search
- OpenAI embeddings for document matching
- Cosine similarity with configurable threshold
- Returns top 5 relevant documents

### AI Chat
- GPT-4 responses
- Real-time streaming
- Tool calling for retrieval and document addition

## Costs

- **Neon**: Free tier (0.5GB storage)
- **Vercel**: Free tier (100GB bandwidth)
- **OpenAI**: Pay-per-use (embeddings ~$0.0001/1K tokens)

## Troubleshooting

### Database Issues
- Check DATABASE_URL format
- Verify Neon database is active
- Ensure SSL mode in connection string

### OpenAI Issues
- Verify API key and credits
- Check rate limits
- Confirm OPENAI_API_KEY is set

### Build Issues
- Run `npm run build` locally first
- Check environment variables in Vercel
- Review function logs for errors

## Contributing

Standard fork/PR workflow. Test locally before submitting.

## License

MIT

---

Built with the Vercel AI SDK
