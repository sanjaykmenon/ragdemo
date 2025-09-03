import { NextResponse } from 'next/server';
import { db } from '@/db';
import { documents } from '@/db/schema';
import OpenAI from 'openai';

const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const sampleFinancialDocuments = [
  {
    title: "Q3 2024 Financial Report",
    content: `QUARTERLY FINANCIAL REPORT - Q3 2024

EXECUTIVE SUMMARY
Our company delivered strong financial performance in Q3 2024, with revenue growing 15% year-over-year to $127.5 million. Net income increased to $18.2 million, representing a 14.3% net margin.

KEY FINANCIAL METRICS
- Total Revenue: $127.5M (15% YoY growth)
- Gross Profit: $89.3M (70% gross margin)
- Operating Income: $22.1M (17.3% operating margin)
- Net Income: $18.2M (14.3% net margin)
- Earnings Per Share: $1.42
- Free Cash Flow: $16.8M

REVENUE BREAKDOWN
- Product Sales: $95.2M (75% of total revenue)
- Subscription Services: $24.1M (19% of total revenue)
- Professional Services: $8.2M (6% of total revenue)

BALANCE SHEET HIGHLIGHTS
- Total Assets: $245.7M
- Cash and Cash Equivalents: $45.3M
- Total Debt: $32.1M
- Shareholders' Equity: $178.4M

OUTLOOK
We expect Q4 2024 revenue to be between $135M-$142M, representing 12-17% year-over-year growth.`,
    metadata: { quarter: "Q3", year: 2024, type: "earnings" }
  },
  {
    title: "Annual Risk Assessment 2024",
    content: `ANNUAL RISK ASSESSMENT REPORT 2024

MARKET RISKS
The company faces several market-related risks that could impact financial performance:

1. COMPETITIVE LANDSCAPE
- Increased competition from both established players and new market entrants
- Pressure on pricing and margins
- Risk of market share erosion

2. ECONOMIC CONDITIONS
- Potential recession could reduce customer demand
- Inflation impacting operational costs
- Interest rate changes affecting borrowing costs

3. REGULATORY RISKS
- Changes in financial regulations
- Data privacy compliance requirements
- Environmental regulations

OPERATIONAL RISKS
- Supply chain disruptions
- Cybersecurity threats
- Key personnel retention
- Technology infrastructure failures

FINANCIAL RISKS
- Credit risk from customer defaults
- Liquidity risk during economic downturns
- Foreign exchange rate fluctuations
- Commodity price volatility

MITIGATION STRATEGIES
- Diversified customer base across industries
- Strong balance sheet with $45.3M in cash
- Comprehensive insurance coverage
- Regular security audits and updates`,
    metadata: { year: 2024, type: "risk-assessment" }
  },
  {
    title: "Cash Flow Analysis - 2024",
    content: `CASH FLOW ANALYSIS 2024

OPERATING CASH FLOW
The company generated strong operating cash flow of $68.4M in 2024, representing a 22% increase from the previous year.

Operating Cash Flow Components:
- Net Income: $72.1M
- Depreciation and Amortization: $12.3M
- Working Capital Changes: $(15.8M)
- Other Operating Activities: $(0.2M)

INVESTING CASH FLOW
Total investing cash flow was $(24.7M), primarily due to:
- Capital Expenditures: $(18.2M)
- Technology Infrastructure: $(8.9M)
- Acquisitions: $(2.1M)
- Asset Disposals: $4.5M

FINANCING CASH FLOW
Financing activities resulted in cash outflow of $(19.3M):
- Debt Repayments: $(12.1M)
- Dividend Payments: $(14.2M)
- Share Repurchases: $(8.5M)
- New Borrowings: $15.5M

FREE CASH FLOW
Free cash flow for 2024 was $50.2M, calculated as:
Operating Cash Flow ($68.4M) minus Capital Expenditures ($18.2M)

LIQUIDITY POSITION
- Cash at Year End: $45.3M
- Available Credit Lines: $25.0M
- Total Liquidity: $70.3M

The company maintains a strong liquidity position to fund operations and growth initiatives.`,
    metadata: { year: 2024, type: "cash-flow" }
  }
];

export async function POST() {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json({ error: 'OpenAI API key not configured' }, { status: 500 });
    }

    const results = [];

    for (const doc of sampleFinancialDocuments) {
      try {
        // Generate embedding for the document
        const embeddingResponse = await openaiClient.embeddings.create({
          model: 'text-embedding-ada-002',
          input: doc.content,
        });

        const embedding = embeddingResponse.data[0].embedding;

        // Insert document into database
        const [insertedDoc] = await db
          .insert(documents)
          .values({
            title: doc.title,
            content: doc.content,
            embedding,
            metadata: JSON.stringify(doc.metadata),
          })
          .returning();

        results.push({
          id: insertedDoc.id,
          title: insertedDoc.title,
          success: true
        });
      } catch (error) {
        console.error(`Error processing document ${doc.title}:`, error);
        results.push({
          title: doc.title,
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return NextResponse.json({
      message: 'Sample documents initialization completed',
      results
    });

  } catch (error) {
    console.error('Error initializing sample documents:', error);
    return NextResponse.json(
      { error: 'Failed to initialize sample documents' },
      { status: 500 }
    );
  }
}