from openai import OpenAI
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
import uuid
from summerize import process_chunks
import openai

sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""

sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

# Provider configuration presets
PROVIDER_PRESETS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_API_BASE_URL",
        "default_model": "gpt-4-1106-preview",
        "embedding_model": "text-embedding-3-small",
    },
    "minimax": {
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_API_BASE_URL",
        "default_base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
        "embedding_model": None,  # MiniMax does not have a public embedding API
    },
}


def _detect_provider():
    """Auto-detect LLM provider from environment variables.

    Priority: LLM_PROVIDER env var > MINIMAX_API_KEY presence > default openai.
    """
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in PROVIDER_PRESETS:
        return explicit
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    return "openai"


def _get_client(provider=None):
    """Build an OpenAI-compatible client for the active provider."""
    provider = provider or _detect_provider()
    preset = PROVIDER_PRESETS[provider]
    api_key = os.getenv(preset["api_key_env"])
    base_url = os.getenv(
        preset["base_url_env"],
        preset.get("default_base_url"),
    )
    return OpenAI(api_key=api_key, base_url=base_url), preset


def _clamp_temperature(temperature, provider):
    """Clamp temperature to valid range for the provider."""
    if provider == "minimax":
        # MiniMax requires temperature in (0.0, 1.0]
        return max(0.01, min(temperature, 1.0))
    return temperature


def get_embedding(text, mod=None):
    provider = _detect_provider()
    preset = PROVIDER_PRESETS[provider]
    if mod is None:
        mod = preset.get("embedding_model") or "text-embedding-3-small"
    if provider == "minimax" and preset.get("embedding_model") is None:
        # Fall back to OpenAI for embeddings when using MiniMax as LLM provider
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
    else:
        client, _ = _get_client(provider)

    response = client.embeddings.create(
        input=text,
        model=mod
    )

    return response.data[0].embedding

def fetch_texts(n4j):
    # Fetch the text for each node
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    # Upload embeddings to Neo4j
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)

    for node in nodes:
        # Calculate embedding for each node's text
        if node['id']:  # Ensure there is text to process
            embedding = get_embedding(node['id'])
            # Store embedding back in the node
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def add_sum(n4j,content,gid):
    sum = process_chunks(content)
    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})
    
    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    return s

def call_llm(sys, user):
    provider = _detect_provider()
    client, preset = _get_client(provider)
    model = os.getenv("LLM_MODEL", preset["default_model"])
    temperature = _clamp_temperature(0.5, provider)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f" {user}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=temperature,
    )
    return response.choices[0].message.content

def find_index_of_largest(nums):
    # Sorting the list while keeping track of the original indexes
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    
    # Extracting the original index of the largest element
    largest_original_index = sorted_with_index[-1][1]
    
    return largest_original_index

def get_response(n4j, gid, query):
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont)
    res = call_llm(sys_prompt_one,user_one)
    user_two = "the question is: " + query + "the last response of it is:" +  res + "the references are: " +  "".join(linkcont)
    res = call_llm(sys_prompt_two,user_two)
    return res

def link_context(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary // Ensure each pair is processed once and exclude "Summary" nodes in relationships
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def merge_similar_nodes(n4j, gid):
    # Define your merge query here. Adjust labels and properties according to your graph schema
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            // Define a threshold for cosine similarity
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.6 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:REFERENCE]->(n)

        // Return results
        RETURN n, m
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result


def str_uuid():
    # Generate a random UUID
    generated_uuid = uuid.uuid4()

    # Convert UUID to a string
    return str(generated_uuid)


