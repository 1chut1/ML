# 🧠 Advanced Database Management Systems (ADBMS) — MongoDB Practicals

---

## 🧩 Practical 1 — CRUD Operations and Querying in MongoDB

### 🎯 **Aim**
Create a MongoDB database with a suitable example and implement:
- Inserting and saving documents (batch insert, insert validation)
- Removing documents
- Updating documents (replacement, modifiers, upserts)
- Demonstrating query criteria and cursor operations

---

### ⚙️ **Concept Overview**

MongoDB is a **NoSQL document-based** database that stores data as **BSON (Binary JSON)**.  
It is schema-less, scalable, and optimized for unstructured data.

| Operation | Description | Equivalent SQL |
|------------|--------------|----------------|
| `insertOne()`, `insertMany()` | Add new documents | `INSERT INTO` |
| `find()`, `findOne()` | Retrieve documents | `SELECT * FROM` |
| `updateOne()`, `updateMany()` | Modify existing documents | `UPDATE SET` |
| `deleteOne()`, `deleteMany()` | Remove documents | `DELETE FROM` |

---

### 💻 **Implementation**

#### Step 1: Create Database and Collection
```bash
use college_db
db.createCollection("teacher_info")
````

#### Step 2: Insert (Batch Insert)

```bash
db.teacher_info.insertMany([
  {teacher_id: "T001", teacher_name: "Priyanka", dept_name: "IT", status: "A", sal: 45000},
  {teacher_id: "T002", teacher_name: "Pradnya", dept_name: "IT", status: "A", sal: 40000},
  {teacher_id: "T003", teacher_name: "Seema", dept_name: "IT", status: "N", sal: 30000}
])
```

#### Step 3: Find and FindOne

```bash
db.teacher_info.find().pretty()
db.teacher_info.findOne({teacher_name: "Priyanka"})
```

#### Step 4: Update Operations

```bash
# Update one document
db.teacher_info.updateOne({teacher_id: "T003"}, {$set: {dept_name: "ETC"}})

# Update multiple documents using a modifier
db.teacher_info.updateMany({status: "A"}, {$inc: {sal: 5000}})

# Upsert example (insert if not found)
db.teacher_info.updateOne(
  {teacher_id: "T010"},
  {$set: {teacher_name: "Neha", dept_name: "MECH", status: "A", sal: 32000}},
  {upsert: true}
)
```

#### Step 5: Remove Documents

```bash
db.teacher_info.deleteOne({teacher_id: "T003"})
```

#### Step 6: Query Demonstrations

```bash
# 1. Retrieve all records
db.teacher_info.find()

# 2. Conditional queries
db.teacher_info.find({sal: {$gt: 40000}})
db.teacher_info.find({sal: {$lt: 40000, $ne: 25000}})

# 3. OR / NOT operators
db.teacher_info.find({$or: [{status: "A"}, {dept_name: "COMP"}]})
db.teacher_info.find({sal: {$not: {$gt: 40000}}})

# 4. Regex query
db.teacher_info.find({teacher_name: {$regex: /^P/}})

# 5. Cursor operations
db.teacher_info.find().sort({sal: -1}).limit(2)
db.teacher_info.find().skip(1).limit(2)
```

---

### 📊 **Result Summary**

| Operation | Method Used                        | Purpose                            |
| --------- | ---------------------------------- | ---------------------------------- |
| Insert    | `insertMany()`                     | Batch document insertion           |
| Update    | `$set`, `$inc`                     | Modify fields and increment salary |
| Delete    | `deleteOne()`                      | Remove specific documents          |
| Query     | `$gt`, `$lt`, `$ne`, `$or`, `$not` | Conditional filters                |
| Cursor    | `limit()`, `sort()`, `skip()`      | Control result display             |

---

### ✅ **Conclusion**

CRUD operations were successfully implemented.
MongoDB efficiently handled insertion, update, deletion, and querying of documents with flexible filters and cursors.

---

## 🧩 Practical 2 — Aggregation Framework and Indexing

### 🎯 **Aim**

Implement the Aggregation Framework and demonstrate creation and deletion of indexes with performance comparison.

---

### ⚙️ **Concept Overview**

* **Aggregation Framework**: Processes multiple documents and returns computed results using stages such as `$match`, `$group`, `$project`, `$sort`, and `$sum`.
* **Indexes**: Improve query performance by allowing MongoDB to search data faster, reducing collection scans.

---

### 💻 **Implementation**

#### Step 1: Create Database and Collection

```bash
use sports_club_db
db.createCollection("users")
```

#### Step 2: Insert Data

```bash
db.users.insertMany([
  { name: "Aarav", join_date: new Date("2024-01-15"), sport: "Cricket" },
  { name: "Bhavna", join_date: new Date("2024-02-10"), sport: "Badminton" },
  { name: "Chirag", join_date: new Date("2024-02-22"), sport: "Football" },
  { name: "Deepika", join_date: new Date("2024-03-01"), sport: "Tennis" },
  { name: "Esha", join_date: new Date("2024-01-25"), sport: "Cricket" }
])
```

#### Step 3: Aggregation Examples

```bash
# 1. Project and Sort
db.users.aggregate([
  { $project: { _id: 0, UPPER_NAME: { $toUpper: "$name" } } },
  { $sort: { UPPER_NAME: 1 } }
])

# 2. Add Computed Field (Month)
db.users.aggregate([
  { $project: { name: 1, monthJoined: { $month: "$join_date" } } },
  { $sort: { monthJoined: 1 } }
])

# 3. Group by Month
db.users.aggregate([
  { $group: { _id: { month: { $month: "$join_date" } }, total_joined: { $sum: 1 } } },
  { $sort: { "_id.month": 1 } }
])
```

#### Step 4: Index Creation and Demonstration

```bash
# Create index
db.users.createIndex({ name: 1 })

# List indexes
db.users.getIndexes()

# Demonstrate index advantage
db.users.find({ name: "Aarav" }).explain("executionStats")

# Drop index
db.users.dropIndex({ name: 1 })
```

---

### 📊 **Result Summary**

| Concept      | Example                      | Description               |
| ------------ | ---------------------------- | ------------------------- |
| Aggregation  | `$project`, `$group`, `$sum` | Summarization of data     |
| Indexing     | `createIndex()`              | Faster queries            |
| Explain Plan | `.explain("executionStats")` | Query performance insight |

---

### ✅ **Conclusion**

The experiment successfully demonstrated MongoDB’s Aggregation Framework for data summarization and the significant performance gain from indexing.

---

## 🧩 Practical 3 — Map-Reduce and Advanced Aggregation

### 🎯 **Aim**

Implement Map-Reduce and Aggregation in MongoDB and demonstrate the use of indexes with examples.

---

### ⚙️ **Concept Overview**

* **Map-Reduce**: Processes large volumes of data by mapping key-value pairs and reducing them to aggregated results.
* **Aggregation**: A modern and optimized alternative to Map-Reduce.
* **Indexing**: Improves performance of frequent query operations.

---

### 💻 **Implementation**

#### Step 1: Create Database and Collection

```bash
use sales_db
db.createCollection("orders")
```

#### Step 2: Insert Documents

```bash
db.orders.insertMany([
  { cust_id: "C001", ord_date: new Date("2024-03-15"), price: 2500 },
  { cust_id: "C002", ord_date: new Date("2024-03-16"), price: 1800 },
  { cust_id: "C001", ord_date: new Date("2024-03-17"), price: 3200 },
  { cust_id: "C003", ord_date: new Date("2024-03-18"), price: 2900 },
  { cust_id: "C002", ord_date: new Date("2024-03-19"), price: 2100 }
])
```

#### Step 3: Implement Map-Reduce

```bash
var mapFunction = function() {
  emit(this.cust_id, this.price);
};

var reduceFunction = function(keyCustId, valuesPrices) {
  return Array.sum(valuesPrices);
};

db.orders.mapReduce(
  mapFunction,
  reduceFunction,
  { out: "total_sales" }
)

db.total_sales.find().pretty()
```

**Output Example:**

```
{ _id: 'C001', value: 5700 }
{ _id: 'C002', value: 3900 }
{ _id: 'C003', value: 2900 }
```

#### Step 4: Aggregation Comparison

```bash
db.orders.aggregate([
  { $group: { _id: "$cust_id", total_sales: { $sum: "$price" } } },
  { $sort: { total_sales: -1 } }
])
```

#### Step 5: Indexing

```bash
# Create index
db.orders.createIndex({ cust_id: 1 })

# Check index usage
db.orders.find({ cust_id: "C001" }).explain("executionStats")

# Drop index
db.orders.dropIndex({ cust_id: 1 })
```

---

### 📊 **Result Summary**

| Operation   | Example                       | Purpose                      |
| ----------- | ----------------------------- | ---------------------------- |
| Map-Reduce  | `emit()`, `Array.sum()`       | Aggregates data by key       |
| Aggregation | `$group`, `$sort`             | Pipeline-based summarization |
| Indexing    | `createIndex()`, `.explain()` | Performance improvement      |

---

### ✅ **Conclusion**

The practical demonstrated:

* The working of Map-Reduce for aggregation.
* Aggregation pipelines as efficient analytical tools.
* The creation, usage, and impact of indexes on performance.

MongoDB’s Aggregation Framework and Indexing techniques enable high-speed analytical operations and optimized data access.

---

## 📚 **References**

* [MongoDB Official Documentation](https://www.mongodb.com/docs/)
* MongoDB Aggregation Framework Manual
* MongoDB CRUD Operations Guide

---

**Prepared by:** Ashwin
**Course:** B.E. IT — Advanced Database Management Systems (ADBMS)
**Tool Used:** MongoDB 8.2, Mongosh 2.5.8

```

---

✅ You can copy the above text **as-is** and save it as  
`ADBMS_MongoDB_Practicals.md` — it’s formatted, complete, and submission-ready.  

Would you like me to add **diagrams placeholders** (like aggregation pipeline, index structure, and Map-Reduce flow) to make it even more presentation-ready?
```
