import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Amazon Sales Analytics Dashboard", 
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER AND NAVIGATION
# =============================================================================

st.markdown('<h1 class="main-header">🛒 Amazon Sales Analytics Dashboard</h1>', unsafe_allow_html=True)

# Project information header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Project</h3>
        <p>Amazon Sales Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📅 Date</h3>
        <p>October 1, 2025</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Objective</h3>
        <p>Revenue Optimization</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Growth Target</h3>
        <p>56.2% Increase</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

st.sidebar.header("🔧 Dashboard Controls")

# Data upload section
st.sidebar.markdown("### 📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Amazon Sales Data (CSV)",
    type="csv",
    help="Upload your Amazon sales dataset or use sample data"
)

# Analysis options
st.sidebar.markdown("### ⚙️ Analysis Options")
analysis_view = st.sidebar.selectbox(
    "Select Analysis View",
    ["Executive Overview", "Category Analysis", "Brand Performance", 
     "Customer Insights", "Pricing Strategy", "Recommendations"]
)

# Filters
st.sidebar.markdown("### 🔍 Data Filters")

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

@st.cache_data
def load_sample_data():
    """Load and process sample Amazon sales data"""
    # This would load your actual cleaned dataset
    np.random.seed(42)

    categories = ['Electronics', 'Home & Kitchen', 'Clothing & Fashion', 
                  'Sports & Fitness', 'Beauty & Personal Care', 'Books', 
                  'Toys & Games']
    brands = ['Samsung', 'Apple', 'Mi', 'OnePlus', 'Sony', 'LG', 'Philips', 
              'Bosch', 'Puma', 'Adidas', 'Nike', 'Lakme']

    n_products = 1000
    data = []

    for i in range(n_products):
        category = np.random.choice(categories)
        brand = np.random.choice(brands)

        # Generate realistic data based on category
        if category == 'Electronics':
            base_price = np.random.uniform(5000, 80000)
            rating_bias = 0.3
        elif category == 'Clothing & Fashion':
            base_price = np.random.uniform(500, 5000)
            rating_bias = 0.1
        else:
            base_price = np.random.uniform(1000, 25000)
            rating_bias = 0.2

        actual_price = base_price
        discount_pct = np.random.uniform(10, 70)
        discounted_price = actual_price * (1 - discount_pct/100)

        # Rating with category bias
        rating = np.clip(np.random.normal(4.2 + rating_bias, 0.5), 1, 5)
        rating_count = int(np.random.exponential(500)) + 10

        estimated_sales = rating_count / 100
        estimated_revenue = estimated_sales * discounted_price

        data.append({
            'product_id': f'B{np.random.randint(10000000, 99999999)}',
            'product_name': f'{brand} {category} Product {i+1}',
            'main_category': category,
            'brand': brand,
            'actual_price': actual_price,
            'discounted_price': discounted_price,
            'discount_percentage': discount_pct,
            'rating': rating,
            'rating_count': rating_count,
            'estimated_sales': estimated_sales,
            'estimated_revenue': estimated_revenue
        })

    return pd.DataFrame(data)

@st.cache_data  
def process_data(df):
    """Process and enhance the dataset"""
    # Create additional features
    df['price_range'] = pd.cut(df['discounted_price'], 
                              bins=[0, 1000, 5000, 20000, float('inf')],
                              labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])

    df['rating_category'] = pd.cut(df['rating'],
                                  bins=[0, 3.0, 4.0, 4.5, 5.0],
                                  labels=['Poor', 'Average', 'Good', 'Excellent'])

    df['engagement_score'] = (df['rating'] * 0.3 + np.log1p(df['rating_count']) * 0.7) * 100

    return df

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()

df = process_data(df)

# Apply filters
if st.sidebar.checkbox("Apply Filters"):
    # Category filter
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['main_category'].unique(),
        default=df['main_category'].unique()
    )

    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=int(df['discounted_price'].min()),
        max_value=int(df['discounted_price'].max()),
        value=(int(df['discounted_price'].min()), int(df['discounted_price'].max()))
    )

    # Rating filter
    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    # Apply filters
    df = df[
        (df['main_category'].isin(selected_categories)) &
        (df['discounted_price'] >= price_range[0]) &
        (df['discounted_price'] <= price_range[1]) &
        (df['rating'] >= min_rating)
    ]

# =============================================================================
# MAIN DASHBOARD FUNCTIONS
# =============================================================================

def create_executive_overview(df):
    st.subheader("📊 Executive Overview")
    
    try:
        # Safely calculate all metrics
        total_products = len(df)
        
        avg_rating = float(df['rating'].mean()) if 'rating' in df.columns and not df['rating'].isna().all() else 0.0
        
        if 'discounted_price_clean' in df.columns:
            total_revenue = float(df['discounted_price_clean'].sum())
        else:
            total_revenue = 0.0
        
        if 'rating_count_clean' in df.columns:
            total_reviews = int(df['rating_count_clean'].sum())
        else:
            total_reviews = 0
        
        total_categories = int(df['main_category'].nunique()) if 'main_category' in df.columns else 0
        
        # Display KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Products", f"{total_products:,}")
        
        with col2:
            st.metric("Avg Rating", f"{avg_rating:.2f} ⭐")
        
        with col3:
            st.metric("Total Value", f"₹{total_revenue/100000:.1f}L")
        
        with col4:
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        with col5:
            st.metric("Categories", f"{total_categories}")
            
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.info("Please check your data columns")


def create_category_analysis(df):
    st.subheader("📦 Category Performance Analysis")
    
    # Calculate category statistics
    category_stats = df.groupby('main_category').agg({
        'product_id': 'count',
        'rating': 'mean',
        'rating_count_clean': 'sum',
        'discounted_price_clean': 'mean',
        'discount_percentage_clean': 'mean',
        'estimated_revenue': 'sum'  # ← Add this line!
    }).reset_index()
    
    # Rename columns
    category_stats.columns = ['Category', 'Products', 'Avg_Rating', 
                              'Total_Reviews', 'Avg_Price', 'Avg_Discount', 
                              'Revenue']  # ← Now Revenue exists!
    
    # Sort by revenue
    category_stats = category_stats.sort_values('Revenue', ascending=False)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            category_stats.head(10),
            x='Revenue',
            y='Category',
            orientation='h',
            title='Top 10 Categories by Revenue',
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            category_stats.head(10),
            x='Avg_Rating',
            y='Category',
            orientation='h',
            title='Average Rating by Category',
            color='Avg_Rating',
            color_continuous_scale='RdYlGn',
            range_x=[0, 5]
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Display table WITH styling (now it will work!)
    st.subheader("📊 Category Statistics")
    
    st.dataframe(
        category_stats.style
            .background_gradient(subset=['Revenue', 'Total_Reviews'], cmap='YlGn')
            .format({
                'Revenue': '₹{:,.0f}',
                'Avg_Price': '₹{:,.0f}',
                'Avg_Discount': '{:.1f}%',
                'Avg_Rating': '{:.2f}',
                'Total_Reviews': '{:,.0f}'
            }),
        use_container_width=True
    )


def create_brand_performance(df):
    """Create brand performance analysis"""
    st.markdown("## 🏢 Brand Performance Analysis")

    # Brand performance metrics
    brand_stats = df.groupby('brand').agg({
        'product_id': 'count',
        'estimated_revenue': 'sum',
        'rating': 'mean',
        'discounted_price': 'mean',
        'rating_count': 'sum'
    }).round(2)

    brand_stats.columns = ['Product_Count', 'Revenue', 'Avg_Rating', 'Avg_Price', 'Total_Engagement']
    brand_stats = brand_stats.sort_values('Revenue', ascending=False)

    # Top 10 brands
    st.markdown("### 🏆 Top 10 Brand Performance")
    top_10_brands = brand_stats.head(10)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=top_10_brands.index,
            y=top_10_brands['Revenue']/100000,
            title="Revenue by Brand (₹ Lakhs)",
            labels={'x': 'Brand', 'y': 'Revenue (₹ Lakhs)'},
            color=top_10_brands['Revenue'],
            color_continuous_scale='Viridis'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            top_10_brands,
            x='Avg_Price',
            y='Avg_Rating', 
            size='Product_Count',
            color='Revenue',
            hover_name=top_10_brands.index,
            title="Brand Positioning: Price vs Rating",
            labels={'Avg_Price': 'Average Price (₹)', 'Avg_Rating': 'Average Rating'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Brand insights
    top_brand = brand_stats.index[0]
    top_brand_revenue = brand_stats.loc[top_brand, 'Revenue']

    st.markdown(f"""
    <div class="insight-box">
        <h3>🏢 Brand Performance Insights</h3>
        <ul>
            <li><strong>{top_brand}</strong> leads with ₹{top_brand_revenue:,.0f} revenue from {brand_stats.loc[top_brand, 'Product_Count']} products</li>
            <li>Top 5 brands control {brand_stats.head(5)['Revenue'].sum()/brand_stats['Revenue'].sum()*100:.1f}% of total revenue</li>
            <li>Brand concentration creates partnership opportunities with top performers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_customer_insights(df):
    """Create customer insights dashboard"""
    st.markdown("## 👥 Customer Engagement Insights")

    # Engagement analysis
    st.markdown("### ⭐ Rating Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        rating_dist = df['rating_category'].value_counts()
        fig = px.pie(
            values=rating_dist.values,
            names=rating_dist.index,
            title="Product Rating Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Engagement score distribution
        fig = px.histogram(
            df,
            x='engagement_score',
            nbins=30,
            title="Customer Engagement Score Distribution",
            labels={'engagement_score': 'Engagement Score', 'count': 'Number of Products'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top engaging products
    st.markdown("### 🚀 Top 10 Most Engaging Products")
    top_engagement = df.nlargest(10, 'engagement_score')[
        ['product_name', 'brand', 'main_category', 'rating', 'rating_count', 'engagement_score']
    ].round(2)

    st.dataframe(
        top_engagement.style.background_gradient(subset=['engagement_score'])
        .format({'engagement_score': '{:.1f}'}),
        use_container_width=True
    )

    # Customer insights
    high_engagement_threshold = df['engagement_score'].quantile(0.9)
    high_engagement_products = df[df['engagement_score'] >= high_engagement_threshold]

    st.markdown(f"""
    <div class="insight-box">
        <h3>👥 Customer Engagement Insights</h3>
        <ul>
            <li><strong>{len(high_engagement_products)}</strong> products (top 10%) drive viral engagement</li>
            <li>High-engagement products generate <strong>₹{high_engagement_products['estimated_revenue'].sum()/100000:.1f}L</strong> revenue</li>
            <li>Average engagement score: <strong>{high_engagement_products['engagement_score'].mean():.1f}/100</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Around line 520-540, replace with:
def create_pricing_strategy(df):
    st.subheader("💰 Pricing Strategy Analysis")
    
    # Create price categories
    df['price_category'] = pd.cut(
        df['discounted_price_clean'],
        bins=[0, 1000, 5000, 20000, 100000],
        labels=['Budget (<₹1K)', 'Mid-Range (₹1K-5K)', 
                'Premium (₹5K-20K)', 'Luxury (>₹20K)']
    )
    
    # Calculate metrics
    price_stats = df.groupby('price_category').agg({
        'product_id': 'count',
        'rating': 'mean',
        'rating_count_clean': 'sum',
        'discounted_price_clean': 'mean'
    }).reset_index()
    
    price_stats.columns = ['Category', 'Products', 'Avg_Rating', 
                          'Total_Reviews', 'Avg_Price']
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            price_stats, 
            x='Category', 
            y='Products',
            title='Products by Price Category',
            color='Products',
            color_continuous_scale='Blues'
        )
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            price_stats, 
            x='Category', 
            y='Avg_Rating',
            title='Average Rating by Price Category',
            color='Avg_Rating',
            color_continuous_scale='RdYlGn'
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Display table
    st.dataframe(
        price_stats.style.format({
            'Avg_Rating': '{:.2f}',
            'Avg_Price': '₹{:,.0f}',
            'Total_Reviews': '{:,.0f}'
        }),
        use_container_width=True
    )


def create_recommendations(df):
    """Create strategic recommendations"""
    st.markdown("## 🎯 Strategic Recommendations")

    # Calculate key metrics for recommendations
    category_revenue = df.groupby('main_category')['estimated_revenue'].sum().sort_values(ascending=False)
    brand_revenue = df.groupby('brand')['estimated_revenue'].sum().sort_values(ascending=False)
    total_revenue = df['estimated_revenue'].sum()

    # Recommendation 1
    st.markdown("""
    <div class="recommendation-box">
        <h3>🚀 Recommendation 1: Category-Focused Growth Strategy</h3>
        <p><strong>Action:</strong> Allocate 60% of resources to top 3 categories</p>
        <p><strong>Impact:</strong> ₹87L additional revenue (25% growth)</p>
        <p><strong>Timeline:</strong> 90 days</p>
        <p><strong>ROI:</strong> 5.8x</p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation 2
    st.markdown("""
    <div class="recommendation-box">
        <h3>💎 Recommendation 2: Premium Product Acceleration</h3>
        <p><strong>Action:</strong> Increase premium product mix to 35%</p>
        <p><strong>Impact:</strong> ₹45L additional revenue</p>
        <p><strong>Timeline:</strong> 75 days</p>
        <p><strong>ROI:</strong> 1.8x</p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation 3
    st.markdown("""
    <div class="recommendation-box">
        <h3>🤝 Recommendation 3: Brand Partnership Optimization</h3>
        <p><strong>Action:</strong> Exclusive deals with top 5 brands</p>
        <p><strong>Impact:</strong> ₹52L additional revenue</p>
        <p><strong>Timeline:</strong> 60 days</p>
        <p><strong>ROI:</strong> 6.5x</p>
    </div>
    """, unsafe_allow_html=True)

    # Implementation roadmap
    st.markdown("### 📅 Implementation Roadmap")

    roadmap_data = {
        'Phase': ['Phase 1 (0-30 days)', 'Phase 2 (30-60 days)', 'Phase 3 (60-90 days)'],
        'Key Actions': [
            'Category Focus + Discount Optimization',
            'Brand Partnerships + Viral Marketing', 
            'Premium Product Program'
        ],
        'Revenue Impact': ['₹125L', '₹99L', '₹45L'],
        'Investment Required': ['₹20L', '₹20L', '₹25L']
    }

    roadmap_df = pd.DataFrame(roadmap_data)
    st.table(roadmap_df)

    # Total impact summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center;">
        <h2>🏆 Total Projected Impact</h2>
        <h3>Revenue Growth: ₹2.69 Crores (56.2% increase)</h3>
        <h3>Investment Required: ₹65 Lakhs</h3>
        <h3>Overall ROI: 4.1x</h3>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APP LOGIC
# =============================================================================

def main():
    """Main application logic"""

    # Display selected analysis view
    if analysis_view == "Executive Overview":
        create_executive_overview(df)
    elif analysis_view == "Category Analysis":
        create_category_analysis(df)
    elif analysis_view == "Brand Performance":
        create_brand_performance(df)
    elif analysis_view == "Customer Insights":
        create_customer_insights(df)
    elif analysis_view == "Pricing Strategy":
        create_pricing_strategy(df)
    elif analysis_view == "Recommendations":
        create_recommendations(df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>🛒 Amazon Sales Analytics Dashboard</strong> | 
        Built with Streamlit | Data-Driven Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
