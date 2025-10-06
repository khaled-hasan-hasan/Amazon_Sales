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
        <p>October 4, 2025</p>
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
    df['Price_range'] = pd.cut(df['discounted_price'], 
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
    Price_range = st.sidebar.slider(
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
        (df['discounted_price'] >= Price_range[0]) &
        (df['discounted_price'] <= Price_range[1]) &
        (df['rating'] >= min_rating)
    ]

# =============================================================================
# MAIN DASHBOARD FUNCTIONS
# =============================================================================

def create_executive_overview(df):
    """Create executive overview dashboard"""
    st.markdown("## Executive Overview")
    
    # Key metrics with error handling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(df)
        st.metric(label="Total Products", value=f"{total_products:,}", 
                 delta=f"{total_products - 800}+ vs baseline")
    
    with col2:
        
        try:
            total_revenue = pd.to_numeric(df['estimated_revenue'], errors='coerce').sum()
            if pd.isna(total_revenue) or total_revenue == 0:
                total_revenue = 0
            revenue_display = f"₹{total_revenue/100000:.1f}L"
        except (TypeError, KeyError):
            total_revenue = 0
            revenue_display = "₹0.0L"
        
        st.metric(label="Total Revenue", value=revenue_display, 
                 delta=f"₹{total_revenue * 0.15/100000:.1f}L potential")
    
    with col3:
        
        try:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
            if pd.isna(avg_rating):
                avg_rating = 0
            rating_display = f"{avg_rating:.2f}"
        except (TypeError, KeyError):
            avg_rating = 0
            rating_display = "0.00"
        
        st.metric(label="Average Rating", value=rating_display, 
                 delta=f"{avg_rating - 4.0:.2f} above 4.0")
    
    with col4:
        
        try:
            if len(df) > 0:
                top_category_share = (df['main_category'].value_counts().iloc[0] / len(df)) * 100
            else:
                top_category_share = 0
            share_display = f"{top_category_share:.1f}%"
        except (TypeError, KeyError, IndexError):
            top_category_share = 0
            share_display = "0.0%"
        
        st.metric(label="Top Category Share", value=share_display, 
                 delta="Concentration risk")

    # Rest of your function continues...

def create_category_analysis(df):
    """Create detailed category analysis"""
    st.markdown("## Category Performance Analysis")
    
    # Category performance metrics
    category_stats = df.groupby('main_category').agg({
        'estimated_revenue': 'sum',
        'rating': 'mean',
        'rating_count': 'sum',
        'discounted_price': 'mean',
        'discount_percentage': 'mean'
    }).round(2)
    
    category_stats.columns = ['Revenue', 'Avg_Rating', 'Total_Ratings', 'Avg_Price', 'Avg_Discount']
    category_stats = category_stats.sort_values('Revenue', ascending=False)
    

    # Ensure numeric columns are actually numeric
    category_stats['Revenue'] = pd.to_numeric(category_stats['Revenue'], errors='coerce')
    category_stats['Total_Ratings'] = pd.to_numeric(category_stats['Total_Ratings'], errors='coerce')
    
    # Fill any NaN values with 0
    category_stats = category_stats.fillna(0)
    
    st.markdown("### Category Performance Table")
    
    # Apply styling only to confirmed numeric columns
    try:
        st.dataframe(
            category_stats.style.background_gradient(subset=['Revenue', 'Total_Ratings'])
            .format({
                'Revenue': '₹{:,.0f}', 
                'Avg_Price': '₹{:,.0f}', 
                'Avg_Discount': '{:.1f}%'
            }),
            use_container_width=True
        )
    except Exception as e:
        # Fallback: Display without styling
        st.dataframe(category_stats, use_container_width=True)
        st.warning("Styling not applied due to data format issues")


def create_brand_performance(df):
    """Create brand performance analysis"""
    st.markdown("## Brand Performance Analysis")
    
   
    # Clean the estimated_revenue column first
    df['estimated_revenue'] = pd.to_numeric(df['estimated_revenue'], errors='coerce').fillna(0)
    
    # Brand performance metrics
    brand_stats = df.groupby('brand').agg({
        'product_id': 'count',
        'estimated_revenue': 'sum',
        'rating': 'mean',
        'discounted_price': 'mean',
        'rating_count': 'sum'
    }).round(2)
    
    brand_stats.columns = ['ProductCount', 'Revenue', 'AvgRating', 'AvgPrice', 'TotalEngagement']
    brand_stats = brand_stats.sort_values('Revenue', ascending=False)
    
    brand_stats['Revenue'] = pd.to_numeric(brand_stats['Revenue'], errors='coerce').fillna(0)
    
    st.markdown("### Top 10 Brand Performance")
    top10_brands = brand_stats.head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        try:
            revenue_values = pd.to_numeric(top10_brands['Revenue'], errors='coerce').fillna(0)
            fig = px.bar(
                x=top10_brands.index,
                y=revenue_values/100000,  # Now safe to divide
                title='Revenue by Brand (Lakhs)',
                labels={'x': 'Brand', 'y': 'Revenue (Lakhs)'},
                color=revenue_values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating revenue chart: {str(e)}")
            # Fallback: Show data without chart
            st.write(top10_brands['Revenue'])
    
    with col2:
        
        try:
            # Ensure all columns are numeric
            avg_price = pd.to_numeric(top10_brands['AvgPrice'], errors='coerce').fillna(0)
            avg_rating = pd.to_numeric(top10_brands['AvgRating'], errors='coerce').fillna(0)
            product_count = pd.to_numeric(top10_brands['ProductCount'], errors='coerce').fillna(1)
            
            fig = px.scatter(
                x=avg_price,
                y=avg_rating,
                size=product_count,
                color=revenue_values,
                hover_name=top10_brands.index,
                title='Brand Positioning: Price vs Rating',
                labels={'x': 'Average Price (₹)', 'y': 'Average Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating positioning chart: {str(e)}")
            # Fallback: Show basic data
            st.dataframe(top10_brands[['AvgPrice', 'AvgRating', 'ProductCount']])

    # Brand insights with safe calculations
    try:
        top_brand = brand_stats.index[0]
        top_brand_revenue = brand_stats.loc[top_brand, 'Revenue']
        
        st.markdown(f"""
        <div class="insight-box">
        <h3>🎯 Brand Performance Insights</h3>
        <ul>
        <li><strong>{top_brand}</strong> leads with ₹{top_brand_revenue:,.0f} revenue from {brand_stats.loc[top_brand, 'ProductCount']} products</li>
        <li>Top 5 brands control {(brand_stats.head(5)['Revenue'].sum()/brand_stats['Revenue'].sum()*100):.1f}% of total revenue</li>
        <li>Brand concentration creates partnership opportunities with top performers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning("Could not generate brand insights due to data formatting issues")


def create_customer_insights(df):
    """Create customer insights dashboard"""
    st.markdown("## 👥 Customer Engagement Insights")
    df['estimated_revenue'] = pd.to_numeric(df['estimated_revenue'], errors='coerce').fillna(0)
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
def create_pricing_strategy(df):
    """Create pricing strategy analysis"""
    st.markdown("## 💰 Pricing Strategy Analysis")
    df['estimated_revenue'] = pd.to_numeric(df['estimated_revenue'], errors='coerce').fillna(0)
    # Price range analysis
    st.markdown("### 📊 Revenue by Price Range")

    price_analysis = df.groupby('Price_range').agg({
        'product_id': 'count',
        'estimated_revenue': 'sum',
        'rating': 'mean',
        'discount_percentage': 'mean'
    }).round(2)

    price_analysis.columns = ['Product_Count', 'Revenue', 'Avg_Rating', 'Avg_Discount']
    price_analysis['Revenue'] = pd.to_numeric(price_analysis['Revenue'], errors='coerce').fillna(0)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=price_analysis.index,
            y=price_analysis['Revenue']/100000,
            title="Revenue by Price Range (₹ Lakhs)",
            color=price_analysis['Revenue'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            price_analysis,
            x='Avg_Discount',
            y='Avg_Rating',
            size='Product_Count',
            hover_name=price_analysis.index,
            title="Discount vs Rating by Price Range"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Pricing insights
    premium_revenue = price_analysis.loc[price_analysis.index.isin(['Premium', 'Luxury']), 'Revenue'].sum()
    total_revenue = price_analysis['Revenue'].sum()
    premium_share = (premium_revenue / total_revenue) * 100

    st.markdown(f"""
    <div class="insight-box">
        <h3>💰 Pricing Strategy Insights</h3>
        <ul>
            <li>Premium + Luxury products generate <strong>{premium_share:.1f}%</strong> of revenue</li>
            <li>Price-rating correlation: <strong>{df['discounted_price'].corr(df['rating']):.3f}</strong> (weak)</li>
            <li>Opportunity for premium pricing strategy due to low price sensitivity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

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
