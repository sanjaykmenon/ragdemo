# Financial Documents RAG Demo

A Next.js application powered by the Vercel AI SDK that uses Retrieval-Augmented Generation (RAG) to answer questions about financial documents. Built with OpenAI's GPT-4 and embeddings, PostgreSQL for vector storage, and a UI with Tailwind CSS and Framer Motion.

## Features

- 🤖 **AI-Powered Chat**: Ask questions about your financial documents using GPT-4
- 📄 **Document Upload**: Upload financial documents (PDF, TXT, CSV, DOCX)
- 🔍 **Semantic Search**: Find relevant information using vector embeddings
- 💾 **Vector Storage**: Store document embeddings in PostgreSQL
- 🎨 **Beautiful UI**: Modern interface with animations and responsive design
- ⚡ **Real-time Streaming**: Live streaming of AI responses
- 🆓 **Free Hosting**: Deploy on Vercel with free PostgreSQL from Neon

## Quick Start

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd ragdemo
npm install
```

### 2. Set up Free PostgreSQL Database (Neon)

1. Go to [Neon](https://neon.tech) and create a free account
2. Create a new project
3. Copy your connection string (it looks like: `postgresql://username:password@host/database?sslmode=require`)

### 3. Get OpenAI API Key

1. Go to [OpenAI](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### 4. Configure Environment Variables

Create a `.env.local` file in the root directory:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required: Neon PostgreSQL Connection String
DATABASE_URL=postgresql://username:password@host/database?sslmode=require

# Optional: For production deployment
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### 5. Set up Database

Generate and run the database migration:

```bash
npm run db:generate
npm run db:migrate
```

### 6. Initialize Sample Data

Start the development server and initialize sample financial documents:

```bash
npm run dev
```

Then in another terminal, initialize sample documents:

```bash
curl -X POST http://localhost:3000/api/init-docs
```

### 7. Start Using the App

1. Open [http://localhost:3000](http://localhost:3000)
2. Try asking questions like:
   - "What are the key financial metrics from Q3 2024?"
   - "What are the main risk factors mentioned?"
   - "Show me information about cash flow"
3. Upload your own financial documents to expand the knowledge base

## Deployment

### Deploy to Vercel (Free)

1. Push your code to GitHub
2. Go to [Vercel](https://vercel.com) and import your repository
3. Add the same environment variables from your `.env.local` file
4. Deploy!

Your app will be live at `https://your-app.vercel.app`

## Project Structure

```
src/
├── app/
│   ├── api/
│   │   ├── chat/route.ts          # Main chat API with RAG
│   │   └── init-docs/route.ts     # Initialize sample documents
│   ├── globals.css                # Global styles
│   ├── layout.tsx                 # Root layout
│   └── page.tsx                   # Main chat interface
├── db/
│   ├── index.ts                   # Database connection
│   └── schema.ts                  # Database schema
drizzle.config.ts                  # Drizzle ORM configuration
package.json                       # Dependencies and scripts
```

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **AI**: Vercel AI SDK with OpenAI GPT-4
- **Database**: PostgreSQL (Neon) with Drizzle ORM
- **Embeddings**: OpenAI text-embedding-ada-002
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **Icons**: Lucide React
- **Hosting**: Vercel (free tier)

## Sample Questions to Try

- "What was the revenue growth in Q3 2024?"
- "Analyze the cash flow trends"
- "What are the main risk factors?"
- "Show me the balance sheet highlights"
- "What's the company's liquidity position?"

## Features in Detail

### Document Upload
- Supports PDF, TXT, CSV, and DOCX files
- Automatically generates embeddings for uploaded content
- Stores documents with metadata in PostgreSQL

### Semantic Search
- Uses OpenAI embeddings to find relevant document sections
- Cosine similarity matching with configurable threshold
- Returns top 5 most relevant documents per query

### AI Chat Interface
- Powered by GPT-4 for intelligent responses
- Real-time streaming of responses
- Tool calling for document retrieval and addition
- Beautiful animated UI with loading states

## Cost Considerations

- **Neon PostgreSQL**: Free tier includes 0.5GB storage, 1 database
- **Vercel Hosting**: Free tier includes 100GB bandwidth, hobby projects
- **OpenAI API**: Pay-per-use (embeddings ~$0.0001/1K tokens, GPT-4 varies)

## Troubleshooting

### Database Connection Issues
- Ensure your DATABASE_URL is correct
- Check that your Neon database is active
- Verify SSL mode is included in connection string

### OpenAI API Issues
- Confirm your API key is valid and has credits
- Check rate limits on your OpenAI account
- Ensure OPENAI_API_KEY environment variable is set

### Build/Deploy Issues
- Run `npm run build` locally to check for errors
- Ensure all environment variables are set in Vercel
- Check Vercel function logs for runtime errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use this project for your own financial document analysis needs!

---

Built with ❤️ using the Vercel AI SDK
