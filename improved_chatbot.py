"""
PRODUCTION HR CHATBOT - CLEAN VERSION
======================================

Silent operation - no verbose debugging
Searches ALL sources: MySQL + Excel + PDF
Always returns an answer
Professional user experience
"""

import os
import re
import hashlib
import mysql.connector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# ============================================
# CONFIG
# ============================================

QDRANT_COLLECTION = "hr_documents"
QDRANT_PATH = "db/qdrant_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

K_CHUNKS = 8  # Increased for better context

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "Vivek@2611"),
    "database": os.getenv("MYSQL_DATABASE", "company_hr"),
}

# Employee detection keywords
EMPLOYEE_KEYWORDS = [
    "employee", "staff", "worker", "emp", "person",
    "salary", "phone", "email", "designation", "department",
]

# Employee ID pattern
EMPLOYEE_ID_PATTERN = r'\b(EMP\d{3,})\b'


# ============================================
# CACHE
# ============================================

class ResponseCache:
    def __init__(self):
        self.cache = {}

    def get(self, question):
        key = hashlib.md5(question.lower().encode()).hexdigest()
        return self.cache.get(key)

    def set(self, question, answer, sources):
        key = hashlib.md5(question.lower().encode()).hexdigest()
        self.cache[key] = (answer, sources)


# ============================================
# SQL HANDLER
# ============================================

class SQLHandler:
    def __init__(self):
        self.conn = None
        self.available = False

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**MYSQL_CONFIG)
            self.available = True
        except:
            self.available = False

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def query(self, question):
        """
        Intent-driven SQL query engine.
        Step 1: Classify intent (count / aggregate / lookup / list)
        Step 2: Extract all entities (department, salary threshold, name, status)
        Step 3: Build parameterized WHERE clause from entities
        Step 4: Execute and format precisely
        """
        if not self.available:
            return None, None

        try:
            q_lower = question.lower()
            intent  = self._classify_intent(q_lower)
            if intent is None:
                return None, None

            entities = self._extract_entities(question, q_lower)

            if intent == "lookup":
                return self._handle_lookup(entities, q_lower)
            elif intent == "count":
                return self._handle_count(entities, q_lower)
            elif intent == "aggregate":
                return self._handle_aggregate(entities, q_lower)
            elif intent == "list":
                return self._handle_list(entities, q_lower)

            return None, None

        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # INTENT CLASSIFIER
    # ------------------------------------------------------------------

    def _classify_intent(self, q_lower):
        """
        Returns one of: 'count' | 'aggregate' | 'lookup' | 'list' | None

        Priority order matters: check most-specific patterns first.
        """
        count_signals     = ["how many", "count", "number of employees",
                             "total employees", "total number"]
        aggregate_signals = ["average salary", "avg salary", "total salary",
                             "highest salary", "lowest salary", "maximum salary",
                             "minimum salary", "max salary", "min salary",
                             "average pay", "mean salary"]
        list_signals      = ["list", "show all", "all employees",
                             "who works", "who are", "employees in",
                             "staff in", "members of"]
        lookup_signals    = ["what is", "what's", "tell me", "give me",
                             "show me", "find", "salary of", "email of",
                             "phone of", "department of"]

        # Aggregates like "average salary" must be checked before generic
        # lookup because they also contain words like "salary"
        for sig in aggregate_signals:
            if sig in q_lower:
                return "aggregate"

        for sig in count_signals:
            if sig in q_lower:
                return "count"

        for sig in list_signals:
            if sig in q_lower:
                return "list"

        # Lookup: a name is present OR a field-keyword is present
        name = self._extract_name(q_lower)
        if name:
            return "lookup"

        for sig in lookup_signals:
            if sig in q_lower:
                return "lookup"

        return None

    # ------------------------------------------------------------------
    # ENTITY EXTRACTOR
    # ------------------------------------------------------------------

    KNOWN_DEPARTMENTS = [
        "hr", "it", "finance", "marketing", "sales",
        "operations", "engineering", "admin", "legal",
    ]
    KNOWN_STATUSES = ["active", "inactive", "terminated", "on leave"]

    def _extract_entities(self, question, q_lower):
        """
        Returns a dict with any of:
          name, first, last, department, status,
          salary_op, salary_val, field
        """
        entities = {}

        # --- Employee name ---
        name = self._extract_name(q_lower)
        if name:
            parts = name.split()
            if len(parts) >= 2:
                entities["name"]  = name
                entities["first"] = parts[0].lower()
                entities["last"]  = " ".join(parts[1:]).lower()

        # --- Department ---
        for dept in self.KNOWN_DEPARTMENTS:
            # Match "HR department", "in HR", "of IT", "IT employees", etc.
            pattern = rf'\b{re.escape(dept)}\b'
            if re.search(pattern, q_lower):
                entities["department"] = dept.upper() if len(dept) <= 3 else dept.title()
                break

        # Catch "department of <name>" edge-case — don't confuse with dept filter
        # (already handled: department entity won't be set for lookup queries)

        # --- Employee status ---
        for status in self.KNOWN_STATUSES:
            if status in q_lower:
                entities["status"] = status.title()
                break

        # --- Salary threshold  e.g. "above 50000", "less than 60,000" ---
        salary_match = re.search(
            r'(above|below|more than|less than|greater than|over|under|at least|at most)\s*[₹]?\s*([\d,]+)',
            q_lower
        )
        if salary_match:
            op_word = salary_match.group(1)
            val_str = salary_match.group(2).replace(",", "")
            entities["salary_val"] = int(val_str)
            entities["salary_op"]  = ">=" if op_word in ("above", "more than", "greater than", "over", "at least") else "<="

        # --- Requested field for lookup ---
        entities["field"] = self._identify_field(q_lower)

        return entities

    # ------------------------------------------------------------------
    # WHERE CLAUSE BUILDER  (shared by count / aggregate / list)
    # ------------------------------------------------------------------

    def _build_where(self, entities, params, exclude_name=False):
        """
        Builds a WHERE clause string + populates params list.
        Returns clause string (empty string if no conditions).
        """
        conditions = []

        if not exclude_name and "first" in entities:
            conditions.append("LOWER(first_name) = %s AND LOWER(last_name) = %s")
            params += [entities["first"], entities["last"]]

        if "department" in entities:
            conditions.append("LOWER(department) = %s")
            params.append(entities["department"].lower())

        if "status" in entities:
            conditions.append("LOWER(status) = %s")
            params.append(entities["status"].lower())

        if "salary_val" in entities:
            op = entities["salary_op"]
            conditions.append(f"salary {op} %s")
            params.append(entities["salary_val"])

        if conditions:
            return "WHERE " + " AND ".join(conditions), params
        return "", params

    # ------------------------------------------------------------------
    # INTENT HANDLERS
    # ------------------------------------------------------------------

    def _handle_count(self, entities, q_lower):
        """Handle: how many employees [in dept / above salary / active]"""
        cursor = self.conn.cursor(dictionary=True)
        params = []
        where, params = self._build_where(entities, params, exclude_name=True)
        cursor.execute(f"SELECT COUNT(*) AS cnt FROM employees {where}", params)
        result = cursor.fetchone()
        count  = result["cnt"]

        # Build a human-readable label for what was counted
        label_parts = []
        if "department" in entities:
            label_parts.append(f"in the {entities['department']} department")
        if "status" in entities:
            label_parts.append(f"with status '{entities['status']}'")
        if "salary_val" in entities:
            op_word = "above" if entities["salary_op"] == ">=" else "below"
            label_parts.append(f"earning {op_word} ₹{entities['salary_val']:,}")

        label = " ".join(label_parts) if label_parts else "in total"
        return f"There are {count} employee(s) {label}.", ["MySQL Database"]

    def _handle_aggregate(self, entities, q_lower):
        """Handle: average / total / highest / lowest salary [in dept / condition]"""
        cursor = self.conn.cursor(dictionary=True)
        params = []
        where, params = self._build_where(entities, params, exclude_name=True)

        # Determine aggregation function
        if any(w in q_lower for w in ["highest", "maximum", "max"]):
            agg, label = "MAX(salary)", "highest salary"
        elif any(w in q_lower for w in ["lowest", "minimum", "min"]):
            agg, label = "MIN(salary)", "lowest salary"
        elif any(w in q_lower for w in ["total", "sum"]):
            agg, label = "SUM(salary)", "total salary"
        else:
            agg, label = "AVG(salary)", "average salary"

        cursor.execute(f"SELECT {agg} AS result FROM employees {where}", params)
        result = cursor.fetchone()
        value  = result["result"] or 0

        scope_parts = []
        if "department" in entities:
            scope_parts.append(f"in the {entities['department']} department")
        if "salary_val" in entities:
            op_word = "above" if entities["salary_op"] == ">=" else "below"
            scope_parts.append(f"earning {op_word} ₹{entities['salary_val']:,}")

        scope = " ".join(scope_parts) if scope_parts else "across all employees"
        return f"The {label} {scope} is ₹{value:,.2f}.", ["MySQL Database"]

    def _handle_list(self, entities, q_lower):
        """Handle: list / show all employees [in dept]"""
        cursor = self.conn.cursor(dictionary=True)
        params = []
        where, params = self._build_where(entities, params, exclude_name=True)
        cursor.execute(
            f"SELECT first_name, last_name, department, designation, salary "
            f"FROM employees {where} ORDER BY first_name LIMIT 20",
            params
        )
        rows = cursor.fetchall()
        if not rows:
            return None, None

        scope = f"in the {entities['department']} department" if "department" in entities else ""
        lines = [f"Employees {scope}:".strip()]
        for r in rows:
            lines.append(
                f"  • {r['first_name']} {r['last_name']} "
                f"({r['designation']}, {r['department']}) — ₹{r['salary']:,.0f}"
            )
        return "\n".join(lines), ["MySQL Database"]

    def _handle_lookup(self, entities, q_lower):
        """Handle: what is [field] of [name]"""
        if "first" not in entities:
            return None, None

        cursor = self.conn.cursor(dictionary=True)
        field  = entities.get("field")

        if field:
            cursor.execute(
                f"SELECT first_name, last_name, {field} FROM employees "
                f"WHERE LOWER(first_name)=%s AND LOWER(last_name)=%s LIMIT 1",
                (entities["first"], entities["last"])
            )
        else:
            cursor.execute(
                "SELECT * FROM employees "
                "WHERE LOWER(first_name)=%s AND LOWER(last_name)=%s LIMIT 1",
                (entities["first"], entities["last"])
            )

        result = cursor.fetchone()
        if result:
            return self._format_result(result, q_lower, field), ["MySQL Database"]
        return None, None

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _extract_name(self, q_lower):
        """Return Title-Cased employee name, or None."""
        # "of Rajesh Kumar" / "for Rajesh Kumar"
        m = re.search(r'\b(?:of|for)\s+([a-z]+(?:\s+[a-z]+)+)', q_lower)
        if m:
            candidate = m.group(1).title()
            # Reject if it looks like a department/keyword phrase
            if not any(d in candidate.lower() for d in self.KNOWN_DEPARTMENTS):
                return candidate

        # "Rajesh Kumar's"
        m = re.search(r'\b([a-z]+\s+[a-z]+)\'s', q_lower)
        if m:
            return m.group(1).title()

        # "employee Rajesh Kumar"
        m = re.search(r'\b(?:employee|person|staff)\s+([a-z]+\s+[a-z]+)', q_lower)
        if m:
            return m.group(1).title()

        return None

    def _identify_field(self, q_lower):
        """Map question keywords → exact DB column name."""
        field_map = {
            "salary":          ["salary", "pay", "earning", "wage", "compensation"],
            "phone":           ["phone", "mobile", "contact number"],
            "email":           ["email", "e-mail", "mail id"],
            "department":      ["department", "dept", "division", "team"],
            "designation":     ["designation", "position", "role", "job title", "title"],
            "city":            ["city", "location", "based in"],
            "status":          ["status", "active or not", "employment status"],
            "date_of_joining": ["date of joining", "joining date", "joined on",
                                "start date", "hire date", "when did", "when joined"],
        }
        for field, keywords in field_map.items():
            if any(kw in q_lower for kw in keywords):
                return field
        return None

    def _format_result(self, result, question, requested_field=None):
        """Format SQL result based on what was requested"""
        name = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
        
        # If a specific field was requested, return only that
        if requested_field:
            value = result.get(requested_field)
            
            if requested_field == 'salary':
                return f"{name} earns ₹{value:,.2f}"
            elif requested_field == 'phone':
                return f"{name}'s phone number is {value}"
            elif requested_field == 'email':
                return f"{name}'s email is {value}"
            elif requested_field == 'department':
                return f"{name} works in the {value} department"
            elif requested_field == 'designation':
                return f"{name}'s designation is {value}"
            elif requested_field == 'city':
                return f"{name} is located in {value}"
            elif requested_field == 'status':
                return f"{name}'s status is {value}"
            elif requested_field == 'date_of_joining':
                return f"{name} joined on {value}"
            else:
                return f"{name}'s {requested_field.replace('_', ' ')}: {value}"
        
        # No specific field - return comprehensive info
        info_parts = []
        
        field_order = ['department', 'designation', 'salary', 'email', 'phone', 'city', 'date_of_joining', 'status']
        
        for field in field_order:
            if field in result and result[field]:
                value = result[field]
                field_name = field.replace('_', ' ').title()
                
                if field == 'salary':
                    info_parts.append(f"Salary: ₹{value:,.2f}")
                else:
                    info_parts.append(f"{field_name}: {value}")
        
        return f"{name}\n" + "\n".join(f"  • {part}" for part in info_parts)


# ============================================
# VECTOR STORE
# ============================================

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.client = QdrantClient(path=QDRANT_PATH)
        self.available = False
        
        try:
            collections = self.client.get_collections().collections
            self.available = any(col.name == QDRANT_COLLECTION for col in collections)
        except:
            self.available = False

    def search_all(self, query, k=8):
        """Search all documents"""
        if not self.available:
            return []
        
        try:
            vector = self.embeddings.embed_query(query)
            results = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=k
            ).points
            
            docs = []
            for r in results:
                docs.append({
                    'content': r.payload.get('content', ''),
                    'source': r.payload.get('source', 'Unknown'),
                    'file_type': r.payload.get('file_type', 'unknown')
                })
            return docs
        except:
            return []

    def search_by_type(self, query, file_type, k=8):
        """Search specific file type"""
        if not self.available:
            return []
        
        try:
            vector = self.embeddings.embed_query(query)
            results = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=k,
                query_filter=Filter(
                    must=[FieldCondition(key="file_type", match=MatchValue(value=file_type))]
                )
            ).points
            
            docs = []
            for r in results:
                docs.append({
                    'content': r.payload.get('content', ''),
                    'source': r.payload.get('source', 'Unknown'),
                    'file_type': r.payload.get('file_type', 'unknown')
                })
            return docs
        except:
            return []


# ============================================
# QUERY ROUTER
# ============================================

def is_employee_query(question, sql_handler=None):
    """
    Returns True if the question should be routed to SQL first.
    Uses the SQL handler's own intent classifier when available.
    """
    if sql_handler:
        return sql_handler._classify_intent(question.lower()) is not None

    # Fallback (if called without handler)
    q_lower = question.lower()
    sql_signals = [
        "how many", "count", "average salary", "avg salary",
        "highest salary", "lowest salary", "list", "show all",
        "employees in", "who works", "salary of", "email of",
        "phone of", "department of",
    ]
    if any(sig in q_lower for sig in sql_signals):
        return True
    if re.search(r'\b([a-z]+\s+[a-z]+)\'s', q_lower):
        return True
    if re.search(r'\b(?:of|for)\s+[a-z]+\s+[a-z]+', q_lower):
        return True
    return False


# ============================================
# ANSWER GENERATION
# ============================================

def get_answer(question, vector_store, llm, cache, sql):
    """
    Smart routing:
    1. Check cache
    2. Try SQL for employee queries
    3. Search vector store (ALL sources)
    4. Generate answer with LLM
    """
    
    # Check cache
    cached = cache.get(question)
    if cached:
        return cached[0], cached[1]

    sources = []
    context = ""

    # ROUTE 1: Employee-specific query → Try SQL first
    if is_employee_query(question, sql):
        sql_answer, sql_sources = sql.query(question)
        if sql_answer:
            cache.set(question, sql_answer, sql_sources)
            return sql_answer, sql_sources

    # ROUTE 2: Search vector store
    # Try all documents first
    docs = vector_store.search_all(question, k=K_CHUNKS)
    
    if docs:
        # Filter by relevance
        keywords = set(question.lower().split())
        relevant_docs = [
            doc for doc in docs
            if any(kw in doc['content'].lower() for kw in keywords)
        ]
        
        if not relevant_docs:
            relevant_docs = docs  # Use all if no keyword match
        
        context = "\n\n".join([doc['content'] for doc in relevant_docs[:K_CHUNKS]])
        sources = list(set([doc['source'] for doc in relevant_docs[:K_CHUNKS]]))

    # ROUTE 3: If no vector results, try Excel specifically
    if not context:
        excel_docs = vector_store.search_by_type(question, "excel", k=K_CHUNKS)
        if excel_docs:
            context = "\n\n".join([doc['content'] for doc in excel_docs])
            sources = list(set([doc['source'] for doc in excel_docs]))

    # Generate answer with LLM
    if context:
        prompt = f"""You are a professional HR assistant for an Indian company.

Context from company documents:
{context}

Question: {question}

Instructions:
- Answer clearly and professionally
- Use Indian Rupee (₹) for money
- Be specific and factual
- If the exact information isn't in context, provide the closest relevant information available
- Never say "information not found" - use what's available

Answer:"""

        try:
            answer = llm.invoke(prompt).content.strip()
            cache.set(question, answer, sources)
            return answer, sources
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response. Please rephrase your question.", []

    # Last resort - no context found
    return "I don't have specific information about that in our current database. Could you rephrase your question or ask about employee records, policies, budgets, or training programs?", []


# ============================================
# INITIALIZATION
# ============================================

def init():
    """Initialize all components silently"""
    
    # Check API key
    if not GROQ_API_KEY or "paste_your" in GROQ_API_KEY:
        raise ValueError("Please configure GROQ_API_KEY")
    
    vector_store = VectorStore()
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2
    )
    
    cache = ResponseCache()
    sql = SQLHandler()
    sql.connect()
    
    # Silent check
    status = []
    if sql.available:
        status.append("SQL")
    if vector_store.available:
        status.append("Documents")
    
    if not status:
        print("⚠️  Warning: No data sources available. Please run ingestion.")
    
    return vector_store, llm, cache, sql


# ============================================
# CHAT INTERFACE
# ============================================

def chat():
    """Clean chat interface"""
    
    try:
        vector_store, llm, cache, sql = init()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "="*60)
    print("HR Assistant")
    print("="*60)
    print("Ask me about employees, policies, budgets, or training.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break

            answer, sources = get_answer(question, vector_store, llm, cache, sql)
            
            print(f"\nBot: {answer}")
            
            if sources:
                source_str = ", ".join(sources)
                print(f"📚 Sources: {source_str}")
            
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    sql.disconnect()


if __name__ == "__main__":
    chat()