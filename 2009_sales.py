import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import datetime as dt
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, DBSCAN

    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as ex
    import plotly.graph_objects as go
    return (
        DBSCAN,
        KMeans,
        MinMaxScaler,
        StandardScaler,
        dt,
        ex,
        go,
        mo,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
        warnings,
    )


@app.cell
def _(pd):
    df = pd.read_csv('online_retail_2009.csv')
    return (df,)


@app.cell
def _(df):
    df_work1 = df.copy()
    return (df_work1,)


@app.cell
def _(df_work1):
    df_work1
    return


@app.cell
def _(mo):
    mo.md("""Check the data for problems - NaNs, neg qty, Invoices and StockCodes""")
    return


@app.cell
def _(df_work1):
    df_work1[df_work1['Customer ID'].isna()].head()
    return


@app.cell
def _(df_work1):
    df_work1[df_work1['Quantity'] < 0.00].head()
    return


@app.cell
def _(df_work1):
    # ID the odd stockcodes

    df_work1['StockCode'] = df_work1['StockCode'].astype('str')

    df_work1[(df_work1['StockCode'].str.match('^\\d{5}$') == False) & (df_work1['StockCode'].str.match('^\\d{5}[a-zA-Z]+$') == False)]['StockCode'].unique()
    return


@app.cell
def _(df_work1):
    df_work1['Invoice'] = df_work1['Invoice'].astype('str')

    df_work1[(df_work1['Invoice'].str.match('^\\d{6}$') == False)]['Invoice'].unique()
    return


@app.cell
def _(df_work1):
    df_work1['Invoice'] = df_work1['Invoice'].astype('str')

    df_work1['Invoice'].str.replace('[0-9]', '', regex=True).unique()
    return


@app.cell
def _(df_work1):
    df_work1[df_work1['Invoice'].str.startswith('A')]
    return


@app.cell
def _(mo):
    mo.md("""Clean up the data""")
    return


@app.cell
def _(df_work1):
    df_clean1 = df_work1.copy()
    return (df_clean1,)


@app.cell
def _(df_clean1):
    # Clean the NaN customers into df_clean1
    df_clean1['Customer ID'] = df_clean1['Customer ID'].astype('str')
    df_clean1.dropna(subset=['Customer ID'], inplace=True)
    return


@app.cell
def _(df_clean1):
    # Check
    df_clean1[df_clean1['Customer ID'].isna()].head()
    return


@app.cell
def _(df_clean1):
    # Clean the invoices into df_clean2

    df_clean1['Invoices'] = df_clean1['Invoice'].astype('str')

    mask1 = (
        df_clean1['Invoice'].str.match('^\\d{6}') == True
    )

    df_clean2 = df_clean1[mask1]

    df_clean2
    return df_clean2, mask1


@app.cell
def _(df_clean2):
    df_clean2[df_clean2['Invoice'].str.endswith('C')]
    return


@app.cell
def _(df_clean2):
    def drop_string_nans(df):
        df = df[df['Customer ID'] != 'nan']
        return df

    df_clean3 = drop_string_nans(df_clean2)
    return df_clean3, drop_string_nans


@app.cell
def _(df_clean3):
    # check results for qty > 0 and price >= 0

    df_clean3.describe()
    return


@app.cell
def _(df_clean3):
    # clean up the stock codes into df_clean4

    df_clean3['StockCode'] = df_clean3['StockCode'].astype('str')

    mask3 = (
        (df_clean3['StockCode'].str.match('^\\d{5}$') == True)
        | (df_clean3['StockCode'].str.match('^\\d{5}[a-zA-Z]+$') == True)
        | (df_clean3['StockCode'].str.match('^PADS$') == True)
    )

    df_clean4 = df_clean3[mask3]
    return df_clean4, mask3


@app.cell
def _(df_clean4):
    # Check stockcodes

    df_clean4[(df_clean4['StockCode'].str.match('^\\d{5}$') == False) & (df_clean4['StockCode'].str.match('^\\d{5}[a-zA-Z]+$') == False)]['StockCode'].unique()
    return


@app.cell
def _(df_clean4):
    df_clean4.dtypes
    return


@app.cell
def _(df_clean4):
    # Round Price to 2 decimals
    df_clean4['Price'] = df_clean4['Price'].round(2)
    return


@app.cell
def _(df_clean4, pd):
    # Set dtype for InvoiceDate to datetime
    df_clean4['InvoiceDate'] = pd.to_datetime(df_clean4['InvoiceDate'])
    return


@app.cell
def _(df_clean4):
    # Drop the .0 from the Customer ID

    df_clean4['Customer ID'] = df_clean4['Customer ID'].str.slice(0, -2)
    return


@app.cell
def _(df_clean4):
    df_clean4.tail(2)
    return


@app.cell
def _(df_clean4):
    df_clean4.rename(columns= {
        'Invoice': 'invoice',
        'StockCode': 'sku',
        'Quantity': 'qty',
        'InvoiceDate': 'date',
        'Price': 'price',
        'Customer ID': 'customer'
    }, inplace=True)
    return


@app.cell
def _(df_clean4):
    # Develop line total price - 'price' * 'qty'

    df_clean4['totalprice'] = df_clean4['qty'] * df_clean4['price']
    df_clean4['totalprice'] = df_clean4['totalprice'].round(2)
    return


@app.cell
def _(df_clean4):
    # Create end state working df

    selected_columns = ['invoice', 'sku', 'qty', 'date', 'price', 'customer', 'totalprice']

    df_cleaned = df_clean4[selected_columns]
    return df_cleaned, selected_columns


@app.cell
def _(df_cleaned):
    df_cleaned
    return


@app.cell
def _(df, df_cleaned):
    # How much data is left after cleaning - 77.33 % is left

    len(df_cleaned) / len(df)
    return


@app.cell
def _(df_cleaned):
    # Aggregate the data into 3 features: Monetary Value; Frequency; Recency

    df_agg = df_cleaned.groupby(by='customer', as_index=False) \
        .agg(
            monetaryvalue = ('totalprice', 'sum'),
            frequency = ('invoice', 'nunique'),
            lastinvoicedate = ('date', 'max')
        )
    return (df_agg,)


@app.cell
def _(df_agg):
    print(df_agg.head())
    return


@app.cell
def _(df_agg):
    max_invoice_date = df_agg['lastinvoicedate'].max()
    return (max_invoice_date,)


@app.cell
def _(df_agg, max_invoice_date):
    df_agg['recency'] = (max_invoice_date - df_agg['lastinvoicedate']).dt.days
    return


@app.cell
def _(df_agg):
    print(df_agg.head())
    return


@app.cell
def _(df_agg, plt, sns):
    # Plot the 3 features

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_agg['monetaryvalue'], color='skyblue')
    plt.title('Monetary Value Boxplot')
    plt.xlabel('Monetary Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_agg['frequency'], color='lightgreen')
    plt.title('Frequency Boxplot')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_agg['recency'], color='salmon')
    plt.title('Recency Boxplot')
    plt.xlabel('Recency')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_agg):
    # The Monetary Value and Frequency featurs are compressed by the number of high value outliers
    # Create new df for the log transform

    selected_columns1 = ['monetaryvalue', 'frequency', 'recency']
    df_agg_log = df_agg[selected_columns1]
    return df_agg_log, selected_columns1


@app.cell
def _(df_agg_log, np):
    # Apply the log transformation

    df_agg_log['monetaryvalue'] = np.log1p(df_agg_log['monetaryvalue'])
    df_agg_log['frequency'] = np.log1p(df_agg_log['frequency'])
    df_agg_log['recency'] = np.log1p(df_agg_log['recency'])
    return


@app.cell
def _(df_agg_log, plt, sns):
    # Plot the transformed data to check results

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_agg_log['monetaryvalue'], color='skyblue')
    plt.title('Monetary Value Boxplot')
    plt.xlabel('Monetary Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_agg_log['frequency'], color='lightgreen')
    plt.title('Frequency Boxplot')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_agg_log['recency'], color='salmon')
    plt.title('Recency Boxplot')
    plt.xlabel('Recency')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_agg_log, plt):
    # This is better. Create a 3-D plot to check scaling

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(projection=('3d'))

    scatter = ax.scatter(df_agg_log['monetaryvalue'], df_agg_log['frequency'], df_agg_log['recency'])

    ax.set_xlabel('Monetary Value')
    ax.set_ylabel(' Frequency')
    ax.set_zlabel('Recency')

    ax.set_title('3-D Scatterplot of Sales Data')
    return ax, fig, scatter


@app.cell
def _(KMeans, df_agg_log, plt, silhouette_score):
    # The scaling in the above plot is reasonable - proceed to KMeans and see what clustering looks like

    max_k = 12
    inertia = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)
        cluster_labels = kmeans.fit_predict(df_agg_log)
        sil_score = silhouette_score(df_agg_log, cluster_labels)
        silhouette_scores.append(sil_score)
        inertia.append(kmeans.inertia_)

    # Plot the inertia and silhouette_scores for values of k

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia, marker='o')
    plt.title('KMeans Inertia for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid=True

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Scores for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid=True

    plt.tight_layout()
    plt.show()
    return (
        cluster_labels,
        inertia,
        k,
        k_values,
        kmeans,
        max_k,
        sil_score,
        silhouette_scores,
    )


@app.cell
def _(KMeans, df_agg_log):
    # The inertia and silhouette score plots support the use of 5 clusters
    # Re-run KMeans for 5 clusters and generate cluster labels for each data line

    kmeans1 = KMeans(n_clusters=5, random_state=42, max_iter=1000)
    cluster_labels1 = kmeans1.fit_predict(df_agg_log)

    cluster_labels1
    return cluster_labels1, kmeans1


@app.cell
def _(cluster_labels1, df_agg_log):
    # Add the cluster labels to the dataframe

    df_agg_log['cluster'] = cluster_labels1
    return


@app.cell
def _(df_agg_log):
    df_agg_log.columns
    return


@app.cell
def _(df_agg_log, plt):
    # 3-D Plot the clusters in colour

    cluster_colors = {0: '#1f77b4', # Blue
                     1: '#ff7f0e', # Orange
                     2: '#2ca02c', # Green
                     3: '#d62728', # Red
                     4: '#A020F0'} # Purple

    colors = df_agg_log['cluster'].map(cluster_colors)

    fig1 = plt.figure(figsize=(8, 8))

    ax1 = fig1.add_subplot(projection=('3d'))

    scatter1 = ax1.scatter(df_agg_log['monetaryvalue'], df_agg_log['frequency'], df_agg_log['recency'], c=colors, marker='o')

    ax1.set_xlabel('Monetary Value')
    ax1.set_ylabel(' Frequency')
    ax1.set_zlabel('Recency')

    ax1.set_title('3-D Scatterplot of Sales Data by Cluster')
    return ax1, cluster_colors, colors, fig1, scatter1


@app.cell
def _(cluster_colors, df_agg_log, plt, sns):
    # The result looks pretty good although there are still some outliers it is likely impossible to mitigate then all
    # Generate violin plots for each cluster 

    plt.figure(figsize=(12, 18))

    plt.subplot(3, 1, 1)
    sns.violinplot(x=df_agg_log['cluster'], y=df_agg_log['monetaryvalue'], palette=cluster_colors, hue=df_agg_log['cluster'])
    sns.violinplot(y=df_agg_log['monetaryvalue'], color='gray', linewidth=1.0)
    plt.title('Monetary Value by Cluster')
    plt.ylabel('Monetary Value')

    plt.subplot(3, 1, 2)
    sns.violinplot(x=df_agg_log['cluster'], y=df_agg_log['frequency'], palette=cluster_colors, hue=df_agg_log['cluster'])
    sns.violinplot(y=df_agg_log['frequency'], color='gray', linewidth=1.0)
    plt.title('Frequency by Cluster')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 3)
    sns.violinplot(x=df_agg_log['cluster'], y=df_agg_log['recency'], palette=cluster_colors, hue=df_agg_log['cluster'])
    sns.violinplot(y=df_agg_log['recency'], color='gray', linewidth=1.0)
    plt.title('Recency by Cluster')
    plt.ylabel('Recency')

    return


if __name__ == "__main__":
    app.run()
