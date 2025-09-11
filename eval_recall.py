#!/usr/bin/env python3

import pandas as pd
import requests
import json
import time
from typing import List, Dict, Any

def load_queries(csv_path: str) -> List[Dict[str, Any]]:
    """Load test queries from CSV file"""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')

def query_search_endpoint(query: str, endpoint_url: str = "http://localhost:8000/api/v1/search") -> Dict[str, Any]:
    """Query the /search endpoint and return results"""
    try:
        payload = {
            "query": query,
            "top_k": 10,  # Get more results to check for target document
            "score_threshold": 0.0
        }
        response = requests.post(endpoint_url, json=payload, timeout=30)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_document_titles_and_content(response_data: Dict) -> List[Dict]:
    """Extract document titles and content from the search response"""
    docs_info = []
    
    if "results" in response_data:
        for result in response_data["results"]:
            docs_info.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "doc_id": result.get("doc_id"),
                "score": result.get("score", 0.0)
            })
    
    return docs_info

def check_expected_span_in_response(docs_info: List[Dict], expected_span: str) -> bool:
    """Check if the expected span appears in the retrieved documents"""
    all_content = ""
    
    # Combine all retrieved content
    for doc in docs_info:
        all_content += doc.get("content", "").lower() + " "
    
    return expected_span.lower() in all_content

def calculate_recall_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate recall metrics from evaluation results"""
    total_queries = len(results)
    successful_queries = [r for r in results if r["success"]]
    
    if not successful_queries:
        return {
            "total_queries": total_queries,
            "successful_queries": 0,
            "top_1_recall": 0.0,
            "top_3_recall": 0.0,
            "top_5_recall": 0.0,
            "mrr": 0.0,
            "span_match_rate": 0.0,
            "by_type": {}
        }
    
    # Calculate recall metrics
    top_1_hits = 0
    top_3_hits = 0  
    top_5_hits = 0
    span_matches = 0
    reciprocal_ranks = []
    
    # Group by type for detailed analysis
    by_type = {}
    
    for result in successful_queries:
        query_type = result["type"]
        if query_type not in by_type:
            by_type[query_type] = {"total": 0, "top_1": 0, "top_3": 0, "top_5": 0, "span_match": 0}
        
        by_type[query_type]["total"] += 1
        
        target_doc = result["target_doc"]
        retrieved_docs = result["retrieved_docs"]
        
        # Find rank of target document
        rank = None
        for i, doc_info in enumerate(retrieved_docs):
            doc_title = doc_info.get("title", "")
            # Check if target document is in the title (remove .md extension)
            target_name = target_doc.replace(".md", "")
            if target_name in doc_title or doc_title in target_name:
                rank = i + 1
                break
        
        # Calculate hits
        if rank:
            if rank == 1:
                top_1_hits += 1
                by_type[query_type]["top_1"] += 1
            if rank <= 3:
                top_3_hits += 1
                by_type[query_type]["top_3"] += 1
            if rank <= 5:
                top_5_hits += 1
                by_type[query_type]["top_5"] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        
        # Check span match
        if result["span_match"]:
            span_matches += 1
            by_type[query_type]["span_match"] += 1
    
    # Calculate final metrics
    total_successful = len(successful_queries)
    
    return {
        "total_queries": total_queries,
        "successful_queries": total_successful,
        "top_1_recall": top_1_hits / total_successful if total_successful > 0 else 0.0,
        "top_3_recall": top_3_hits / total_successful if total_successful > 0 else 0.0,
        "top_5_recall": top_5_hits / total_successful if total_successful > 0 else 0.0,
        "mrr": sum(reciprocal_ranks) / total_successful if total_successful > 0 else 0.0,
        "span_match_rate": span_matches / total_successful if total_successful > 0 else 0.0,
        "by_type": by_type
    }

def run_evaluation():
    """Run the complete RAG recall evaluation"""
    
    print("🚀 Starting RAG Recall Evaluation")
    print("=" * 50)
    
    # Load test queries
    queries = load_queries("rag_recall_testpack_v1/queries.csv")
    print(f"📋 Loaded {len(queries)} test queries")
    
    # Run evaluation
    results = []
    
    for i, query_data in enumerate(queries, 1):
        query_id = query_data["qid"]
        query_text = query_data["query"]
        target_doc = query_data["target_doc"]
        expected_span = query_data["expected_span"]
        query_type = query_data["type"]
        
        print(f"\n🔍 Query {i}/{len(queries)} (ID: {query_id})")
        print(f"   Text: {query_text}")
        print(f"   Target: {target_doc}")
        print(f"   Type: {query_type}")
        
        # Query the endpoint
        response = query_search_endpoint(query_text)
        
        if response["success"]:
            response_data = response["data"]
            retrieved_docs = extract_document_titles_and_content(response_data)
            span_match = check_expected_span_in_response(retrieved_docs, expected_span)
            
            doc_titles = [doc["title"] for doc in retrieved_docs]
            print(f"   ✓ Retrieved docs: {doc_titles[:3]}...")  # Show first 3
            print(f"   ✓ Span match: {span_match}")
            
            results.append({
                "qid": query_id,
                "query": query_text,
                "target_doc": target_doc,
                "expected_span": expected_span,
                "type": query_type,
                "success": True,
                "retrieved_docs": retrieved_docs,
                "span_match": span_match,
                "response_data": response_data
            })
        else:
            print(f"   ✗ Query failed: {response['error']}")
            results.append({
                "qid": query_id,
                "query": query_text,
                "target_doc": target_doc,
                "expected_span": expected_span,
                "type": query_type,
                "success": False,
                "error": response["error"]
            })
        
        # Brief pause to avoid overwhelming the API
        time.sleep(0.5)
    
    # Calculate metrics
    print(f"\n📊 Calculating Metrics")
    print("=" * 50)
    
    metrics = calculate_recall_metrics(results)
    
    # Display results
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Successful Queries: {metrics['successful_queries']}")
    print(f"")
    print(f"📈 Recall Metrics:")
    print(f"  Top-1 Recall: {metrics['top_1_recall']:.3f} ({metrics['top_1_recall']*100:.1f}%)")
    print(f"  Top-3 Recall: {metrics['top_3_recall']:.3f} ({metrics['top_3_recall']*100:.1f}%)")
    print(f"  Top-5 Recall: {metrics['top_5_recall']:.3f} ({metrics['top_5_recall']*100:.1f}%)")
    print(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.3f}")
    print(f"  Span Match Rate: {metrics['span_match_rate']:.3f} ({metrics['span_match_rate']*100:.1f}%)")
    
    # Display by type
    print(f"\n📊 Performance by Query Type:")
    for query_type, type_metrics in metrics["by_type"].items():
        total = type_metrics["total"]
        if total > 0:
            print(f"  {query_type} ({total} queries):")
            print(f"    Top-1: {type_metrics['top_1']/total:.3f}")
            print(f"    Top-3: {type_metrics['top_3']/total:.3f}")
            print(f"    Top-5: {type_metrics['top_5']/total:.3f}")
            print(f"    Span Match: {type_metrics['span_match']/total:.3f}")
    
    # Save results
    results_file = "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return metrics, results

if __name__ == "__main__":
    run_evaluation()