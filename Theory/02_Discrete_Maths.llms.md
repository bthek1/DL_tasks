# Discrete Maths

Discrete Mathematics is the study of **mathematical structures that are countable or otherwise distinct and separable**. It forms the foundation of computer science, cryptography, logic, and many other fields. One of its core branches is **Boolean Algebra**, which is used in digital logic design, computer architecture, and programming.

Author

Benedict Thekkel

## **1. Introduction to Discrete Mathematics**

Discrete Mathematics involves the study of: - **Finite or countable structures** (e.g., integers, graphs, logical statements) - **Non-continuous objects** (unlike calculus or real analysis) - Applications in **computer science, cryptography, and algorithm design**.

### **Key Areas of Discrete Mathematics:**

1.  **Logic and Propositional Calculus**\
2.  **Set Theory**\
3.  **Combinatorics**\
4.  **Graph Theory**\
5.  **Number Theory**\
6.  **Boolean Algebra**\
7.  **Algorithms and Complexity**

------------------------------------------------------------------------

## **2. Logic and Propositional Calculus**

**Logic** is the study of reasoning and truth. In discrete math, we deal with **propositional logic** and **predicate logic**.

### **2.1. Propositions and Truth Values**

- **Proposition:** A declarative statement that is either **true (T)** or **false (F)**.
  - Example:
    - ( P: ) → True\
    - ( Q: ) → False

### **2.2. Logical Connectives**

| Symbol | Name          | Meaning        | Example   |
|--------|---------------|----------------|-----------|
| `¬`    | Negation      | Not            | ( ¬P )    |
| `∧`    | Conjunction   | And            | ( P ∧ Q ) |
| `∨`    | Disjunction   | Or             | ( P ∨ Q ) |
| `⇒`    | Implication   | If… then…      | ( P ⇒ Q ) |
| `⇔`    | Biconditional | If and only if | ( P ⇔ Q ) |

### **2.3. Truth Tables**

Truth tables show the truth value of logical expressions.

**Example: Truth Table for Implication (`P ⇒ Q`)**

| ( P ) | ( Q ) | ( P ⇒ Q ) |
|-------|-------|-----------|
| T     | T     | T         |
| T     | F     | F         |
| F     | T     | T         |
| F     | F     | T         |

### **2.4. Logical Equivalences**

Logical statements can be simplified using equivalences: - **De Morgan’s Laws:**\
- ( ¬(P ∧ Q) ≡ ¬P ∨ ¬Q )\
- ( ¬(P ∨ Q) ≡ ¬P ∧ ¬Q )

- **Double Negation:**
  - ( ¬(¬P) ≡ P )
- **Implication Law:**
  - ( P ⇒ Q ≡ ¬P ∨ Q )

------------------------------------------------------------------------

## **3. Set Theory**

**Set Theory** is the study of collections of distinct objects.

### **3.1. Basic Notation and Operations**

- **Set Notation**: ( A = {1, 2, 3} )
- **Element of a set**: ( x ∈ A )
- **Subset**: ( A ⊆ B )
- **Empty Set**: ( ∅ )

### **3.2. Set Operations**

| Operation | Symbol | Meaning | Example |
|----|----|----|----|
| Union | ( ∪ ) | Elements in either set | ( A ∪ B ) |
| Intersection | ( ∩ ) | Elements in both sets | ( A ∩ B ) |
| Difference | ( A  B ) | Elements in A but not in B | ( A = {1, 2}, B = {2, 3} ), ( A  B = {1} ) |
| Complement | ( A^c ) | Elements not in set A | ( A^c ) |
| Cartesian Product | ( A × B ) | Ordered pairs | ( A = {1, 2}, B = {x, y} ), ( A × B = {(1, x), (1, y), (2, x), (2, y)} ) |

------------------------------------------------------------------------

## **4. Combinatorics**

**Combinatorics** is the study of counting, arrangements, and combinations.

### **4.1. Basic Counting Principles**

- **Addition Principle**: If event A can occur in ( m ) ways and event B can occur in ( n ) ways, then either A or B can occur in ( m + n ) ways.
- **Multiplication Principle**: If event A can occur in ( m ) ways and for each way of A, event B can occur in ( n ) ways, then the events can occur in ( m × n ) ways.

### **4.2. Permutations and Combinations**

| Type            | Formula       | Example                            |
|-----------------|---------------|------------------------------------|
| **Permutation** | ( P(n, r) = ) | Arrangements of 3 out of 5 letters |
| **Combination** | ( C(n, r) = ) | Choosing 2 out of 4 items          |

### **4.3. Binomial Theorem**

- **Formula**:\
  ( (x + y)^n = ∑\_{k=0}^{n} C(n, k) x^{n-k} y^k )
- **Example**:\
  ( (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3 )

------------------------------------------------------------------------

## **5. Boolean Algebra**

**Boolean Algebra** is a branch of algebra dealing with **binary values (0 and 1)**.

### **5.1. Basic Boolean Operations**

| Operation | Symbol    | Meaning      | Example                  |
|-----------|-----------|--------------|--------------------------|
| **NOT**   | ( ¬A )    | Inversion    | ( ¬1 = 0, ¬0 = 1 )       |
| **AND**   | ( A ∧ B ) | Logical AND  | ( 1 ∧ 1 = 1, 1 ∧ 0 = 0 ) |
| **OR**    | ( A ∨ B ) | Logical OR   | ( 1 ∨ 0 = 1, 0 ∨ 0 = 0 ) |
| **XOR**   | ( A ⊕ B ) | Exclusive OR | ( 1 ⊕ 1 = 0, 1 ⊕ 0 = 1 ) |

### **5.2. Boolean Laws and Identities**

- **Identity Laws**:
  - ( A ∧ 1 = A )\
  - ( A ∨ 0 = A )
- **Domination Laws**:
  - ( A ∧ 0 = 0 )\
  - ( A ∨ 1 = 1 )
- **Idempotent Laws**:
  - ( A ∧ A = A )\
  - ( A ∨ A = A )
- **Complement Laws**:
  - ( A ∧ ¬A = 0 )\
  - ( A ∨ ¬A = 1 )
- **De Morgan’s Laws**:
  - ( ¬(A ∧ B) = ¬A ∨ ¬B )\
  - ( ¬(A ∨ B) = ¬A ∧ ¬B )

### **5.3. Boolean Functions and Truth Tables**

**Example: ( F = A ∧ (B ∨ ¬C) )**

| ( A ) | ( B ) | ( C ) | ( ¬C ) | ( B ∨ ¬C ) | ( A ∧ (B ∨ ¬C) ) |
|-------|-------|-------|--------|------------|------------------|
| 0     | 0     | 0     | 1      | 1          | 0                |
| 0     | 0     | 1     | 0      | 0          | 0                |
| 1     | 1     | 0     | 1      | 1          | 1                |
| 1     | 0     | 1     | 0      | 0          | 0                |

------------------------------------------------------------------------

Back to top
