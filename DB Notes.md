# 🧠 Advanced Database Management Systems (ADBMS) — MongoDB Practicals (Theoretical Notes)

---

## 🧩 Practical 1 — CRUD Operations and Querying in MongoDB

### 🎯 **Aim**
To create and manage a MongoDB database by performing **basic CRUD operations**:
- Insertion and saving of documents
- Deletion and updating of records
- Querying data using various conditions and cursor operations

---

### 📘 **Theory**

MongoDB is a **NoSQL document-oriented database** that stores data in the form of **collections** and **documents** rather than tables and rows.

- A **collection** is similar to a table in SQL.  
- A **document** is analogous to a record and is stored in **BSON (Binary JSON)** format.

#### **1. CRUD Operations Overview**
CRUD stands for:
- **Create** → Adding documents to a collection  
- **Read** → Retrieving documents from a collection  
- **Update** → Modifying existing documents  
- **Delete** → Removing documents from a collection  

Each of these operations corresponds to familiar SQL operations but with far greater flexibility in MongoDB since schema enforcement is optional.

#### **2. Data Model**
MongoDB is **schema-less**, meaning:
- Each document can have a different structure.
- Fields can be added dynamically.
- Relationships between data are often represented using **embedding** or **referencing**.

#### **3. Query Operators**
MongoDB provides operators for flexible filtering:
- **Comparison:** `$eq`, `$ne`, `$gt`, `$lt`, `$gte`, `$lte`  
- **Logical:** `$or`, `$and`, `$not`  
- **Element:** `$exists`, `$type`  
- **Regex:** Pattern matching using regular expressions  

#### **4. Cursor Operations**
When performing `find()` operations, MongoDB returns a **cursor** — a pointer to the result set.  
Common cursor methods:
- `limit(n)` → restricts number of results
- `skip(n)` → skips specific documents
- `sort({field: 1 or -1})` → sorts data ascending or descending  

#### **5. Use Cases**
CRUD operations are foundational in every MongoDB application:
- Data ingestion
- Simple analytics
- Application backends for dynamic data
- Logging or configuration databases

---

### 📗 **Conclusion**
MongoDB’s CRUD operations provide flexibility for schema-less data manipulation.  
Through simple commands and operators, complex queries can be performed efficiently without rigid table structures, making MongoDB suitable for modern web and analytics applications.

---

## 🧩 Practical 2 — Aggregation Framework and Indexing

### 🎯 **Aim**
To study and implement **MongoDB’s Aggregation Framework** and **Indexing techniques** to summarize, transform, and optimize database queries.

---

### 📘 **Theory**

#### **1. Aggregation Framework**
Aggregation in MongoDB is analogous to **SQL’s GROUP BY** with more flexibility.  
It allows processing of data through a **pipeline**, where each stage transforms the documents and passes the results to the next stage.

##### **Key Aggregation Stages**
| Stage | Description |
|--------|--------------|
| `$match` | Filters documents (similar to `WHERE`) |
| `$group` | Groups documents by field(s) and performs aggregation operations (`$sum`, `$avg`, `$max`, `$min`) |
| `$project` | Reshapes documents, includes or excludes fields |
| `$sort` | Orders documents based on one or more fields |
| `$limit` and `$skip` | Restricts and paginates output |
| `$lookup` | Performs joins between collections |
| `$count` | Counts documents after filters |

##### **Example Use-Cases**
- Finding total sales per region  
- Counting users joined per month  
- Computing average or maximum values  

The **aggregation pipeline** is powerful because:
- It’s optimized internally for performance.
- It can replace Map-Reduce in most modern use cases.

---

#### **2. Indexing in MongoDB**

**Indexing** improves query performance by creating a data structure that allows MongoDB to locate documents quickly rather than scanning the entire collection.

##### **Types of Indexes**
| Index Type | Description |
|-------------|--------------|
| **Single Field Index** | Created on one field; speeds up queries filtering that field. |
| **Compound Index** | Created on multiple fields; efficient for multi-criteria queries. |
| **Text Index** | Supports text search within string fields. |
| **Hashed Index** | Enables sharding and distributed data access. |

##### **Advantages of Indexes**
- Reduces query execution time.
- Minimizes the number of documents scanned.
- Enables efficient sorting and range queries.
- Supports uniqueness enforcement.

##### **Disadvantages**
- Requires additional disk space.
- Slows down insert/update operations slightly because indexes must be updated.

##### **Explain Plan**
MongoDB provides `.explain("executionStats")` to analyze query efficiency.  
- **Without Index:** Collection scan (`COLLSCAN`)  
- **With Index:** Index scan (`IXSCAN`) — faster and optimized  

---

### 📗 **Conclusion**
The aggregation framework simplifies data transformation and summarization, while indexing ensures efficient data retrieval.  
Together, they form the core of MongoDB’s performance-oriented architecture, enabling fast analytical and transactional processing.

---

## 🧩 Practical 3 — Map-Reduce and Advanced Aggregation

### 🎯 **Aim**
To implement **Map-Reduce** and **Aggregation operations** in MongoDB and demonstrate indexing benefits.

---

### 📘 **Theory**

#### **1. Map-Reduce Concept**

Map-Reduce is a **programming paradigm** for processing large datasets.  
It consists of two main steps:
- **Map phase:** Processes each document and emits a key-value pair.
- **Reduce phase:** Aggregates values with the same key to compute a final result.

##### **Working Mechanism**
1. **Map Function:**  
   Extracts key-value pairs from each document.  
   Example: `{cust_id: "C001", amount: 2000}` → emit(`C001`, 2000)
2. **Reduce Function:**  
   Aggregates all emitted values for each key, e.g., total purchase amount per customer.
3. **Output Collection:**  
   The results are stored in a new collection for further use.

##### **Advantages**
- Handles large-scale data (parallelizable)
- Highly customizable aggregation logic
- Suitable for batch analytics

##### **Disadvantages**
- Slower than Aggregation Pipeline
- Deprecated in modern MongoDB versions (replaced by `$group`, `$accumulator`, etc.)

---

#### **2. Aggregation Framework vs. Map-Reduce**

| Feature | Aggregation Framework | Map-Reduce |
|----------|------------------------|------------|
| Execution Model | Pipeline-based | JavaScript-based functions |
| Performance | Faster and optimized | Slower (interpreted JS) |
| Use Case | Most data summaries | Complex, custom analytics |
| Output | In-memory or temporary | Stored as collection |

Modern MongoDB prefers **aggregation pipelines** since they are faster, more concise, and optimized internally.

---

#### **3. Indexing Review**
Indexes can also improve Map-Reduce and Aggregation queries by minimizing disk I/O and allowing efficient key-based lookups.

---

### 📗 **Conclusion**
Map-Reduce and Aggregation operations in MongoDB demonstrate how data processing and summarization can be achieved within the database engine itself, reducing the need for external computation tools.  
With indexing, these operations become scalable and performant for both transactional and analytical workloads.

---

## 🧩 **Comparative Overview**

| Concept | Purpose | Key Functions | Performance Benefit |
|----------|----------|----------------|---------------------|
| CRUD | Data creation and manipulation | `insert`, `update`, `find`, `delete` | Fundamental data handling |
| Aggregation | Data summarization and transformation | `$group`, `$project`, `$sum`, `$sort` | Fast in-memory computations |
| Indexing | Query optimization | `createIndex()`, `.explain()` | Reduces search time |
| Map-Reduce | Key-value based aggregation | `map()`, `reduce()` | Complex analytical processing |

---

## 🧠 **Key Takeaways**
- MongoDB’s document model allows flexible, schema-less data representation.  
- CRUD operations enable dynamic interaction with collections.  
- The Aggregation Framework provides structured analytics capabilities inside the database.  
- Indexes significantly improve query performance.  
- Map-Reduce demonstrates the ability of MongoDB to handle large-scale data processing.

---


