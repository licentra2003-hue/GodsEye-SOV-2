import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORTED FOR CORS FIX
from pydantic import BaseModel, Field
from supabase import Client, create_client

# --- CONFIGURATION ---
load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GodsEye-Matrix")

# Initialize Clients
try:
    if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
        raise ValueError("Missing API Keys in environment variables.")
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Initialization Failed: {e}")
    raise e

# --- DATA MODELS (PHASE 1) ---

class SingleQueryInsight(BaseModel):
    original_analysis_id: str = Field(description="The UUID of the source row being analyzed")
    query_text: str = Field(description="The search query text")
    sov_score: int = Field(description="0-99 score. Strict scoring based on visibility.")
    category_relevance: str = Field(description="High, Medium, or Low relevance.")
    citation_status: str = Field(description="Cited, Mentioned, or None")
    winning_entity: str = Field(description="The brand/domain that owns the most 'Real Estate' in the answer")
    reasoning: str = Field(description="One sentence explaining the score.")

class BatchAnalysisResult(BaseModel):
    insights: List[SingleQueryInsight]

class CalculationRequest(BaseModel):
    snapshot_id: str
    engine: str
    debug: bool = False

# --- INTELLIGENCE LAYER ---

class MatrixCalculator:
    def __init__(self, snapshot_id: str, engine: str, debug_mode: bool = False):
        self.snapshot_id = snapshot_id
        self.engine = engine.lower().strip()
        self.debug_mode = debug_mode
        self.model = genai.GenerativeModel('gemini-2.5-pro') # Use Flash for speed/cost
        
        self.product_data = {}
        self.product_name = "Unknown"
        self.snapshot_meta = {}

    def _clean_json_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r"\n```$", "", cleaned)
        return cleaned.strip()

    async def _analyze_chunk(self, chunk_data: List[Dict]) -> List[SingleQueryInsight]:
        """Phase 1: Strict Scoring with ANTI-HALLUCINATION Rules."""
        prod_context = f"""
        **Product:** {self.product_name}
        **Desc:** {self.product_data.get('description', 'N/A')}
        """

        prompt = f"""
        You are an AEO Auditor.
        **Target:** {prod_context}
        **Engine:** {self.engine.upper()}
        **Task:** Score {len(chunk_data)} search results.
        
        **CRITICAL SCORING RULES (STRICT ADHERENCE REQUIRED):**
        1. **Strict Identity Check:** You are scoring the visibility of **"{self.product_name}"** ONLY.
        2. **The "Zero" Rule:** If "{self.product_name}" is NOT explicitly mentioned in the text or links, the SOV Score MUST be **0**.
           - **DO NOT** score the "Winning Entity". If AppsFlyer is mentioned but "{self.product_name}" is not, your score for "{self.product_name}" is 0.
        3. **Winning Entity:** Name the brand that actually dominates the answer (e.g., AppsFlyer, HubSpot).
        4. **Relevance:** High/Medium/Low based on query intent match.
        
        **Input Data:**
        {json.dumps(chunk_data, default=str)}
        """

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BatchAnalysisResult
                )
            )
            clean = self._clean_json_text(response.text)
            # Note: Depending on Pydantic version, use model_validate_json or parse_raw
            try:
                result = BatchAnalysisResult.model_validate_json(clean)
            except AttributeError:
                result = BatchAnalysisResult.parse_raw(clean)
            
            # Safety clamp
            for i in result.insights:
                if i.sov_score >= 100: i.sov_score = 99
            return result.insights
        except Exception as e:
            logger.error(f"Gemini Chunk Error: {e}")
            return []

    async def _build_intelligence_context(self, insights: List[Dict]) -> Dict:
        """Phase 2: Context Analysis with Invisibility Handling."""
        if not insights: return {}
        
        # Limit payload for context generation
        sample = insights[:50] if len(insights) > 50 else insights

        # --- UPDATE: Added 'executive_summary' request to prompt ---
        prompt = f"""
        Analyze these {len(sample)} AI Search results for the Client Product: "{self.product_name}".
        
        **CRITICAL INSTRUCTION:**
        - The Client ("{self.product_name}") might be invisible (Score 0). 
        - If they are absent, your analysis must focus on **WHY** they are absent (Gap Analysis).
        - **DO NOT** shift the focus to the "Winning Entity" (e.g., do not write patterns about AppsFlyer as if they were the client).
        - Analyze Competitors only as **Threats**.

        **Required JSON Output:**
        Return a JSON object with these exact keys:
        1. "headlines": A list of short strings (e.g., ["Threat: Competitor X Dominance", "Gap: Missing Keyword Y"]).
        2. "executive_summary": A 2-sentence high-level narrative explaining the product's visibility, the main competitor threat, and the most critical gap. Be direct and professional.
        
        Data: {json.dumps(sample, default=str)}
        """
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            return json.loads(self._clean_json_text(response.text))
        except Exception as e:
            logger.error(f"Context Generation Failed: {e}")
            return {"headlines": [], "executive_summary": "AI Analysis failed."}

    async def _extract_generative_dna(self, rows: List[Dict], query_col: str) -> Dict:
        """Phase 3: The Generative DNA Extraction."""
        dna_input_data = []
        for r in rows:
            raw_data = r.get('raw_serp_results', {})
            # Handle variable field names
            ai_text = raw_data.get('ai_overview_text') or raw_data.get('answer') or raw_data.get('text')
            
            if ai_text and len(ai_text) > 50:
                dna_input_data.append({
                    "query": r.get(query_col),
                    "ai_response_text": ai_text
                })

        if not dna_input_data:
            return {"status": "No AI text available for DNA extraction"}

        # Limit DNA analysis to avoid token limits (e.g., top 10 longest responses)
        dna_sample = sorted(dna_input_data, key=lambda x: len(x['ai_response_text']), reverse=True)[:10]

        prompt = f"""
        Extract the **"Generative DNA"** from these {self.engine.upper()} search results.
        
        **Goal:** Create a robust knowledge bank that explains how the AI thinks about this topic.
        
        **Required JSON Structure:**
        1. **Contextual Logic:** Map "Triggers" -> "Context" (e.g., "Query about pricing -> Context is ROI").
        2. **Intent Mappings:** "User Problem" -> "AI Solution".
        3. **Hard Data Bank:** List specific **immutable facts** found in the text (Entities, Stats, Standards).
        4. **Knowledge Gaps:** What sub-topics are mentioned but not explained?
        
        **Input Data:**
        {json.dumps(dna_sample, default=str)}
        """

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            return json.loads(self._clean_json_text(response.text))
        except Exception as e:
            logger.error(f"DNA Extraction Error: {e}")
            return {"error": str(e)}

    async def run(self):
        try:
            logger.info(f"🚀 Processing Snapshot {self.snapshot_id}")

            # --- STEP 1: LOAD METADATA ---
            snap_res = supabase.table("analysis_snapshots").select("*").eq("id", self.snapshot_id).execute()
            if not snap_res.data: return {"success": False, "error": "Snapshot ID not found"}
            self.snapshot_meta = snap_res.data[0]
            
            prod_res = supabase.table("products").select("*").eq("id", self.snapshot_meta['product_id']).execute()
            self.product_data = prod_res.data[0]
            self.product_name = self.product_data.get('product_name')

            # --- STEP 2: FETCH ALL SCRAPED DATA ---
            source_table = "product_analysis_perplexity" if self.engine == 'perplexity' else "product_analysis_google"
            query_col = "optimization_prompt" if self.engine == 'perplexity' else "search_query"

            raw_res = supabase.table(source_table)\
                .select(f"id, {query_col}, raw_serp_results")\
                .eq("snapshot_id", self.snapshot_id)\
                .execute()
            
            all_scraped_rows = raw_res.data
            if not all_scraped_rows: return {"success": False, "error": "No scraped data found."}

            # --- STEP 3: SMART UPSERT (CALCULATE DELTA) ---
            scraped_ids = [r['id'] for r in all_scraped_rows]
            
            # Fetch existing insights only for these IDs
            existing_insights_res = supabase.table("sov_query_insights")\
                .select("*")\
                .in_("analysis_id", scraped_ids)\
                .execute()
            
            existing_map = {row['analysis_id']: row for row in existing_insights_res.data}
            
            # The Delta: Rows we haven't analyzed yet
            new_rows_to_analyze = [r for r in all_scraped_rows if r['id'] not in existing_map]
            
            logger.info(f"📊 Delta Report: Total={len(all_scraped_rows)}, Existing={len(existing_map)}, New Work={len(new_rows_to_analyze)}")

            # --- STEP 4: ANALYZE NEW ROWS ONLY ---
            new_insights_objects = [] 
            
            if new_rows_to_analyze:
                BATCH_SIZE = 15
                for i in range(0, len(new_rows_to_analyze), BATCH_SIZE):
                    chunk = new_rows_to_analyze[i:i+BATCH_SIZE]
                    formatted = [
                        {
                            "original_analysis_id": r['id'], 
                            "query": r.get(query_col), 
                            "raw_serp_data": r.get('raw_serp_results')
                        } 
                        for r in chunk
                    ]
                    
                    chunk_insights = await self._analyze_chunk(formatted)
                    new_insights_objects.extend(chunk_insights)
                
                # IMMEDIATE SAVE: Write new insights to DB
                if new_insights_objects:
                    insights_payload = [
                        {
                            "analysis_id": i.original_analysis_id,
                            "product_id": self.snapshot_meta['product_id'],
                            "engine": self.engine,
                            "sov_score": i.sov_score,
                            "category_relevance": i.category_relevance,
                            "citation_status": i.citation_status,
                            "winning_source": i.winning_entity,
                            "ai_narrative": i.reasoning
                        }
                        for i in new_insights_objects
                    ]
                    
                    # Batch Insert
                    c_size = 50
                    for i in range(0, len(insights_payload), c_size):
                        supabase.table("sov_query_insights").insert(insights_payload[i:i+c_size]).execute()

            # --- STEP 5: MERGE OLD + NEW FOR GLOBAL MATH ---
            # Convert NEW objects to dicts
            new_insights_dicts = [
                {
                    "sov_score": i.sov_score,
                    "category_relevance": i.category_relevance,
                    "citation_status": i.citation_status,
                    "ai_narrative": i.reasoning,
                    "query_text": next((r[query_col] for r in new_rows_to_analyze if r['id'] == i.original_analysis_id), "Unknown")
                }
                for i in new_insights_objects
            ]

            # Convert OLD DB rows to dicts
            old_insights_dicts = [
                {
                    "sov_score": row['sov_score'],
                    "category_relevance": row['category_relevance'],
                    "citation_status": row['citation_status'],
                    "ai_narrative": row['ai_narrative'],
                    "query_text": next((r[query_col] for r in all_scraped_rows if r['id'] == row['analysis_id']), "Unknown")
                }
                for row in existing_map.values()
            ]

            full_dataset = old_insights_dicts + new_insights_dicts
            total_items = len(full_dataset)
            
            if total_items == 0:
                return {"success": True, "message": "No data available to analyze yet."}

            # --- STEP 6: CALCULATE AGGREGATES ---
            total_sov = sum(i['sov_score'] for i in full_dataset)
            avg_sov = round(total_sov / total_items)
            
            rel_map = {'high': 95, 'medium': 50, 'low': 0}
            avg_rel = round(sum(rel_map.get(str(i['category_relevance']).lower(), 0) for i in full_dataset) / total_items)
            
            cited_count = sum(1 for i in full_dataset if str(i['citation_status']).lower() == 'cited')
            citation_score = round((cited_count / total_items) * 100)

            # --- STEP 7: RE-GENERATE CONTEXT & DNA ---
            # We re-run Context on the full set because "The Story" changes as more data comes in
            # We re-run DNA on the full set to ensure it captures everything
            
            context_task = self._build_intelligence_context(full_dataset)
            dna_task = self._extract_generative_dna(all_scraped_rows, query_col)
            
            context_patterns, generative_dna = await asyncio.gather(context_task, dna_task)

            # Narrative Generation (UPDATED)
            # Try to get the AI summary, fallback to stats if missing
            ai_summary = context_patterns.get("executive_summary")
            
            if ai_summary:
                narrative = ai_summary
            else:
                # Fallback to hardcoded stats if AI fails
                narrative = f"Based on {total_items} queries. Visibility is {avg_sov}%. Trust Score is {citation_score}%."
                if avg_sov < 30 and avg_rel > 70:
                    narrative += f" Critical Gap: {self.product_name} is highly relevant but virtually invisible."

            # --- STEP 8: UPSERT PRODUCT SNAPSHOT ---
            existing_sov_snap = supabase.table("sov_product_snapshots")\
                .select("id")\
                .eq("snapshot_id", self.snapshot_id)\
                .eq("engine", self.engine)\
                .execute()

            summary_payload = {
                "snapshot_id": self.snapshot_id,
                "product_id": self.snapshot_meta['product_id'],
                "batch_id": self.snapshot_meta.get('batch_id'),
                "engine": self.engine,
                "global_sov_score": avg_sov,
                "citation_score": citation_score,
                "category_relevance": avg_rel,
                "total_queries_analyzed": total_items,
                "narrative_summary": narrative,
                "context_patterns": context_patterns,
                "scraped_generative_dna": generative_dna, # New Field
                "analyzed_at": "now()" 
            }

            if existing_sov_snap.data:
                # UPDATE existing row
                supabase.table("sov_product_snapshots")\
                    .update(summary_payload)\
                    .eq("id", existing_sov_snap.data[0]['id'])\
                    .execute()
            else:
                # INSERT new row
                supabase.table("sov_product_snapshots").insert(summary_payload).execute()

            return {
                "success": True,
                "data": {
                    "sov": avg_sov,
                    "total_analyzed": total_items,
                    "newly_processed": len(new_rows_to_analyze),
                    "dna_status": "Extracted" if generative_dna else "Failed"
                }
            }

        except Exception as e:
            logger.error(f"Matrix Calc Error: {e}")
            return {"success": False, "error": str(e)}

# --- FASTAPI ---
app = FastAPI()

# --- ADD CORS MIDDLEWARE HERE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/calculate-sov')
async def handle_calculate_sov(payload: CalculationRequest):
    try:
        calculator = MatrixCalculator(payload.snapshot_id, payload.engine, payload.debug)
        # Directly await the async method, avoiding blocking behavior
        result = await calculator.run()
        
        if result.get('success'):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown Error'))

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # 'workers' parameter doesn't work well with programmatic launch in this context,
    # but Uvicorn handles async concurrency naturally in a single process.
    uvicorn.run(app, host='0.0.0.0', port=port)
