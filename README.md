# Optimizing Inventory Management through Product Classification with K-MEANS, K-MEANS++, and K-MEDOIDS 📊📈

## Introduction
This project analyzes and optimizes inventory management using machine learning clustering algorithms such as K-Means, K-Means++, and K-Medoids. The goal is to improve inventory turnover and reduce excess stock through data-driven insights.

## Contents

1. **Source Code**
   - **Python**: The source code for data analysis and machine learning models is implemented in Python. This includes notebooks and scripts for data preprocessing, clustering algorithms (K-Means, K-Means++, K-Medoids), and visualization.
     - **Files:**
       - `Fullcode.ipynb`: Contains the complete Python code for exploratory data analysis, data pre-processing, determining the optimal number of clusters, standard K-Means clustering, K-Means++ clustering, K-Medoids clustering, and evaluating the clustering performance.
    
2. **Power BI Files**
   - `Product_dashboard.pbix`: This dashboard displays stock quantity over time and analyzes inventory levels by product category.
   - `Sale_dashboard.pbix`: This dashboard focuses on analyzing sales quantity and total cost, as well as sales quantity by product category.
   - `Kmeans_plus_insight_dashboard.pbix`: This dashboard provides insights from K-Means++ clustering, including analysis of product clusters based on sales quantity, stock quantity, and total cost.
       
3. **Report Files**
   - **Files:** `235MI7101_Group6_FinalProject.docx`, `235MI7101_Group6_FinalProject.pdf`
   - **Description:** The study utilizes K-Means, K-Means++, and K-Medoids clustering algorithms to classify products, aiming to enhance inventory management. By analyzing data from e-commerce platforms, the research demonstrates that clustering can reduce holding costs, prevent overstocking, and improve decision-making processes in inventory control​​​​​​.

4. **Datasets**
   - **Data Source:**
     - Looker Ecommerce BigQuery Dataset from [Kaggle.com](https://www.kaggle.com/datasets/mustafakeser4/looker-ecommerce-bigquery-dataset?fbclid=IwZXh0bgNhZW0CMTAAAR2_ZUOvZ-82FVKHHkn-z1GfBxC_tZ_D15G5-gUaj7iAv8lekVpyKzZv26s_aem_CLkHKGJwHhXB4-EgF0NjBg&select=order_items.csv).
     - **Files:**
         - `distribution_centers.csv`: Contains information about distribution centers including ID, name, latitude, and longitude. ​
         - `events.csv`
         - `inventory_items.csv`
         - `order_items.csv`: Contains details of items in each order, including order ID, product ID, status, and sale price.
         - `orders.csv`: Contains order information including order ID, user ID, status, and timestamps for order events.
         - `products.csv`: Contains product information including cost, category, name, brand, and distribution center ID.
         - `users.csv`: Contains user details including personal information, address, and traffic source.
   - **Cleaned Data:**
     - `Data_model.csv`
       - **Description:** Contains detailed product information such as product ID, name, category, sales quantity, and stock quantity. The data is used for analyzing and optimizing inventory management.
     - `Clustering_data.csv`
       - **Description:** This data supports product clustering to optimize inventory management using clustering algorithms.

5. **Gantt Chart**
   - The Gantt chart outlines the project timeline from June 25, 2024, to July 30, 2024, covering phases from idea research, data preparation, analysis, and model evaluation to conclusion and presentation preparation, with all tasks completed on schedule.    
   
## Further Information

**Contact Person for Questions:**  
   - Name: Nguyen Thi Ngoc Giau 
   - Email: giauntn21411ca@st.uel.edu.vn

**Include Credits**
   - Leader: Nguyen Thi Ngoc Giau - K214111970
   - Member:
     + Nguyen Trung Hieu Hien - K214111971
     + Nguyen Thi Thu Ngan - K214111974
     + Nguyen Trinh Thao Nghi - K214111975
     + Tran Duc Anh Duy - K214110859

**Keywords:** Optimize Inventory, Inventory Management, Clustering Algorithms, E-commerce, K-Means, K-Means++, K-Medoids

**Language:** English

**Date of Data Collection:** July, 2024
