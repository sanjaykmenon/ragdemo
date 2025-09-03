# Financial Documents RAG Demo

Next.js app for querying financial documents using RAG. Uses GPT-4 for chat, OpenAI embeddings for search, and PostgreSQL for vector storage.

## What it does

- Chat interface for asking questions about financial docs
- Document upload (PDF, TXT, CSV, DOCX)
- Vector similarity search with embeddings
- PostgreSQL storage with pgvector extension
- Streaming responses
- Deploy on Vercel + Neon (both free tiers)

## Setup

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd ragdemo
npm install
```

### 2. Database (Neon PostgreSQL)

1. Create account at [Neon](https://neon.tech)
2. Create new project
3. Copy connection string: `postgresql://username:password@host/database?sslmode=require`

### 3. OpenAI API Key

1. Get key from [OpenAI](https://platform.openai.com/api-keys)
2. Copy the key (starts with `sk-`)

### 4. Environment Variables

Create `.env.local`:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
DATABASE_URL=postgresql://username:password@host/database?sslmode=require

# Optional
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### 5. Database Setup

```bash
npm run db:generate
npm run db:migrate
```

### 6. Load Sample Data

```bash
npm run dev
```

In another terminal:

```bash
curl -X POST http://localhost:3000/api/init-docs
```

### 7. Usage

1. Open [http://localhost:3000](http://localhost:3000)
2. Try questions like:
   - "What are the Q3 2024 financial metrics?"
   - "What are the main risk factors?"
   - "Show me cash flow information"
3. Upload your own documents

## Deployment

### Vercel

1. Push to GitHub
2. Import in [Vercel](https://vercel.com)
3. Add environment variables
4. Deploy

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── chat/route.ts          # Chat API with RAG
│   │   └── init-docs/route.ts     # Sample doc loader
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

## Example Queries

- "What was the revenue growth in Q3 2024?"
- "Analyze the cash flow trends"
- "What are the main risk factors?"
- "Show me the balance sheet highlights"

## How it works

### Document Upload
- Supports PDF, TXT, CSV, DOCX
- Generates embeddings automatically
- Stores in PostgreSQL with metadata

### Search
- OpenAI embeddings for document matching
- Cosine similarity with 0.7 threshold
- Returns top 5 relevant documents

### Chat
- GPT-4 responses
- Streaming output
- Tool calling for retrieval and document addition

## Costs

- **Neon**: Free tier (0.5GB storage)
- **Vercel**: Free tier (100GB bandwidth)
- **OpenAI**: Pay-per-use (~$0.0001/1K tokens for embeddings)

## Troubleshooting

### Database Issues
- Check DATABASE_URL format
- Verify Neon database is running
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

Fork, make changes, submit PR. Test locally first.

## License

MIT
