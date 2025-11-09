# 🧠 Advanced Database Management Systems (ADBMS) — MongoDB Practicals

---

## 🧩 Practical 1 — CRUD Operations and Querying in MongoDB

### 🎯 **Aim**
To create a MongoDB database and perform basic operations like adding, viewing, updating, and deleting records.

---

### 📘 **Theory**

MongoDB is a **NoSQL database** that stores data as **documents** in **collections** instead of rows and tables like SQL.  
Each document is written in **BSON (Binary JSON)** format and can have a flexible structure.

CRUD stands for:
- **C**reate → Add data  
- **R**ead → Get or view data  
- **U**pdate → Change existing data  
- **D**elete → Remove data  

---

### ⚙️ **Explanation of Operations**

#### **1. Create (Insert Data)**
Used to add data into a collection.  
Example: Adding multiple teachers at once with `insertMany()` adds several records together to save time.

#### **2. Read (View Data)**
Used to get documents using `find()` or `findOne()`.  
We can filter results using conditions like greater than (`$gt`) or less than (`$lt`).  
Example: `find({sal: {$gt:40000}})` gives teachers earning more than 40,000.

#### **3. Update (Modify Data)**
Used to change data in existing documents.  
- `$set` changes a field value.  
- `$inc` increases a number field.  
Example: `updateMany({status:"A"}, {$inc:{sal:5000}})` increases salary by 5000 for all active teachers.

#### **4. Delete (Remove Data)**
Used to remove documents using `deleteOne()` or `deleteMany()`.  
Example: `deleteOne({teacher_id:"T003"})` removes one teacher record.

#### **5. Query and Cursor**
Queries help to filter and display data.  
- `.limit(n)` → show limited records  
- `.sort({field:1})` → sort ascending  
- `.skip(n)` → skip certain results  

Example:  
`find().sort({sal:-1}).limit(2)` shows top two highest-paid teachers.

---

### 📙 **Key Query Operators**

| Type | Operator | Meaning |
|------|-----------|----------|
| Comparison | `$gt`, `$lt`, `$eq`, `$ne` | greater than, less than, equal, not equal |
| Logical | `$or`, `$and`, `$not` | combine or negate conditions |
| Regex | `$regex` | search patterns in strings |

---

### ✅ **Conclusion**
CRUD operations are the basic functions in MongoDB.  
They make it easy to add, change, remove, and view data from collections in a flexible way.

---

## 🧩 Practical 2 — Aggregation Framework and Indexing

### 🎯 **Aim**
To learn how to summarize data using the **Aggregation Framework** and improve search speed using **Indexes** in MongoDB.

---

### 📘 **Theory**

#### **1. Aggregation Framework**
Aggregation means collecting and summarizing data.  
In MongoDB, data is processed step-by-step using a **pipeline** where each step changes or filters the data.

##### **Common Steps in Aggregation**
- `$match` → filters data (like SQL WHERE)  
- `$group` → groups data and uses functions like `$sum`, `$avg`, `$max`, `$min`  
- `$project` → selects specific fields  
- `$sort` → arranges results  
- `$limit` and `$skip` → control number of outputs  

Example:  
Grouping users by sport to count how many play each sport.

**Why it’s useful:**  
It helps perform calculations (like totals or averages) directly inside the database without external tools.

---

#### **2. Indexing**
An **index** is like a book index — it helps MongoDB find data faster.  
Without an index, MongoDB checks every document (**collection scan**).  
With an index, it can directly go to the required data (**index scan**).

##### **Common Index Types**
| Type | Use |
|------|-----|
| Single Field | Simple search (e.g., name) |
| Compound | Multiple fields (e.g., region + income) |
| Text Index | For text searching |
| Hashed Index | For distributed data |

##### **Advantages**
- Faster search results  
- Useful for sorting and filtering  
- Less CPU work for queries  

##### **Disadvantages**
- Uses more disk space  
- Slightly slows insert/update operations because the index must also be updated  

##### **Example Explanation**
Creating an index on `name` →  
`createIndex({name:1})` makes searching by name much faster.

Using `.explain("executionStats")` helps compare query speed before and after using indexes.

---

### ✅ **Conclusion**
The aggregation framework helps in analyzing and summarizing data, while indexing makes queries run much faster.  
Both features make MongoDB efficient for real-time data and analytics.

---

## 🧩 Practical 3 — Map-Reduce and Advanced Aggregation

### 🎯 **Aim**
To understand **Map-Reduce** and how it compares with the **Aggregation Framework**, and to see how indexing improves performance.

---

### 📘 **Theory**

#### **1. What is Map-Reduce?**
Map-Reduce is a two-step process used for analyzing large amounts of data.

##### **Steps:**
1. **Map phase:**  
   Reads each document and emits a key-value pair.  
   Example: For each order, emit (customer_id, price).
2. **Reduce phase:**  
   Combines all values with the same key to calculate totals or averages.  
   Example: Add up all prices per customer to get total spending.

The result is stored in a new collection.

##### **Why it’s useful:**
- Can handle very large data.  
- Good for custom calculations.  
- Works like data grouping, but more flexible.

##### **Why it’s less used now:**
- Slower than Aggregation.  
- Uses JavaScript functions, which are slower than built-in stages.

---

#### **2. Aggregation vs Map-Reduce**

| Feature | Aggregation | Map-Reduce |
|----------|--------------|------------|
| Method | Step-by-step pipeline | JavaScript functions |
| Speed | Faster | Slower |
| Use | Common analytics | Complex calculations |
| Output | Inline or temporary | New collection |

Aggregation is now preferred because it’s faster and easier to use.

---

#### **3. Role of Indexing**
Indexes also help Map-Reduce and Aggregation.  
For example, adding an index on `cust_id` before grouping customer totals makes the query run faster since MongoDB skips unnecessary documents.

---

### ✅ **Conclusion**
Map-Reduce and Aggregation are both used to process and analyze data in MongoDB.  
Aggregation is faster and simpler, while Map-Reduce is better for custom, complex logic.  
Indexing helps both by reducing time to find and process data.

---

## 🧩 Summary Comparison

| Concept | Purpose | Key Function | Benefit |
|----------|----------|--------------|----------|
| CRUD | Data creation and modification | insert, find, update, delete | Basic data handling |
| Aggregation | Data summarization | `$group`, `$project`, `$sort` | Fast analysis |
| Indexing | Faster searching | `createIndex()` | Quicker query results |
| Map-Reduce | Custom large-scale analysis | `map()`, `reduce()` | Handles complex processing |

---

## 🧠 Key Points
- MongoDB stores data in flexible, JSON-like documents.  
- CRUD is used for everyday data handling.  
- Aggregation is used for analysis like totals or averages.  
- Indexing speeds up search and filtering.  
- Map-Reduce handles big data analysis but is slower than Aggregation.
