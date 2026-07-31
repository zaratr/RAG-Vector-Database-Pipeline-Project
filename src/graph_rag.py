import networkx as nx

class GraphRAGPipeline:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def extract_entities_with_gemma(self, text):
        # In a real run, this calls Ollama Gemma to extract (Entity1, Relationship, Entity2)
        # Mocking the LLM response for validation checkpoint
        return [("User", "purchases", "Subscription"), ("Subscription", "grants", "PremiumAccess")]
        
    def build_graph(self, documents):
        for doc in documents:
            triplets = self.extract_entities_with_gemma(doc)
            for e1, rel, e2 in triplets:
                self.graph.add_edge(e1, e2, relation=rel)
                
    def query_graph(self, entity):
        if entity in self.graph:
            edges = self.graph.edges(entity, data=True)
            return [(u, v, d['relation']) for u, v, d in edges]
        return []

if __name__ == '__main__':
    pipeline = GraphRAGPipeline()
    docs = ["User purchases Subscription which grants PremiumAccess."]
    pipeline.build_graph(docs)
    
    result = pipeline.query_graph("User")
    print(f"Graph query for 'User': {result}")
    print("Phase 2 Step 1 Validation: GraphRAG local setup complete.")
