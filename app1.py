import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="PhonePe Expense Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# RESET FUNCTIONALITY
# ---------------------------------------------------------
def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------
# SIDEBAR DESIGN & CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 PhonePe Expense Dashboard")
    st.markdown("---")
    
    # Developer Branding Section
    st.markdown("### 👨‍💻 Developer Profile")
    st.markdown("**Ankit Kumar Ojha**")
    
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        st.markdown(
            '<a href="mailto:ankitojha1184@gmail.com" target="_blank" title="Email Me">'
            '<img src="https://img.icons8.com/color/36/000000/gmail-new.png"/></a>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<a href="https://www.linkedin.com/in/ankit-kumar-ojha-94835b323" target="_blank" title="LinkedIn Profile">'
            '<img src="https://img.icons8.com/color/36/000000/linkedin.png"/></a>',
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("### ⚙️ Dashboard Controls")
    
    # Transaction Type Filter
    txn_type_filter = st.radio(
        "Filter Transactions:",
        ["All", "Expenses Only (Debit)", "Income Only (Credit)"],
        index=0
    )
    
    st.markdown("---")
    # Working Reset Button
    if st.button("🔄 Reset App", on_click=reset_app):
        pass
        
    st.markdown("---")
    st.caption("© 2026 PhonePe Expense Dashboard v1.0")

# ---------------------------------------------------------
# LOAD TRAINED MODEL AND VECTORIZER
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('transaction_classifier.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

try:
    model, tfidf = load_artifacts()
except Exception as e:
    st.sidebar.error("❌ Failed to load model files. Ensure .pkl files exist.")

# ---------------------------------------------------------
# MAIN DASHBOARD CONTENT
# ---------------------------------------------------------
st.title("📊 PhonePe Expense Dashboard")
st.markdown("Upload your raw PhonePe transaction CSV file to classify expenses and view detailed monthly reports.")

# Smart CSV Loader Function
def load_clean_csv(uploaded_file):
    lines = [uploaded_file.readline().decode('utf-8', errors='ignore') for _ in range(20)]
    uploaded_file.seek(0)
    
    header_row = 0
    for idx, line in enumerate(lines):
        if "transaction details" in line.lower() or "amount" in line.lower() or "date" in line.lower():
            header_row = idx
            break
            
    df = pd.read_csv(uploaded_file, skiprows=header_row, on_bad_lines='skip')
    return df

# File Uploader with Key binding for Reset
uploaded_file = st.file_uploader("Upload Transaction Statement (CSV)", type=['csv'], key="file_uploader")

if uploaded_file is not None:
    try:
        df = load_clean_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        # Apply Sidebar Debit/Credit Filter safely
        type_col = next((c for c in df.columns if 'type' in c.lower() or 'credit/debit' in c.lower()), None)
        if type_col:
            if txn_type_filter == "Expenses Only (Debit)":
                df = df[df[type_col].astype(str).str.lower().str.contains('debit')]
            elif txn_type_filter == "Income Only (Credit)":
                df = df[df[type_col].astype(str).str.lower().str.contains('credit')]

        # Raw Data Preview (Safely remove only unwanted columns)
        cols_to_hide_preview = ['Transaction Details', 'Transaction ID', 'UTR', 'Credit/debit instrument']
        cols_present = [c for c in cols_to_hide_preview if c in df.columns]
        
        if len(cols_present) < len(df.columns):
            preview_df = df.drop(columns=cols_present)
        else:
            preview_df = df
            
        st.subheader("📋 Raw Data Preview")
        st.dataframe(preview_df.head(), use_container_width=True)
        
        if st.button("Run Classification & Analytics", type="primary"):
            target_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ["detail", "description", "payee", "particulars", "remark"]):
                    target_col = col
                    break
            
            if target_col is None:
                st.error("Could not find 'Transaction Details' column in CSV.")
            else:
                # Text Preprocessing & Model Prediction
                text_data = df[target_col].fillna("").astype(str).str.lower().str.strip()
                vec_data = tfidf.transform(text_data)
                df['Predicted_Category'] = model.predict(vec_data)
                
                # Filter out unwanted columns for display
                cols_to_remove = ['Transaction ID', 'UTR', 'Credit/debit instrument', 'Instructions', 'Instruction']
                display_df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
                
                st.success("⚡ Classification Completed Successfully!")
                
                # Dashboard Tabs
                tab1, tab2, tab3 = st.tabs(["📋 Classified Data", "📈 Category Analytics", "🗓️ Month-Wise Report"])
                
                with tab1:
                    st.subheader("Classified Transactions")
                    st.dataframe(display_df, use_container_width=True)
                    
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Categorized CSV",
                        data=csv,
                        file_name='categorized_transactions.csv',
                        mime='text/csv',
                    )
                    
                with tab2:
                    st.subheader("Category-wise Spending Breakdown")
                    amt_col = next((c for c in df.columns if 'amount' in c.lower()), None)
                    if amt_col:
                        df[amt_col] = pd.to_numeric(df[amt_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                        
                        cat_summary = df.groupby('Predicted_Category')[amt_col].sum().reset_index()
                        cat_summary = cat_summary.sort_values(by=amt_col, ascending=False)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.dataframe(cat_summary, use_container_width=True)
                        with col2:
                            st.bar_chart(data=cat_summary, x='Predicted_Category', y=amt_col)
                    else:
                        st.warning("'Amount' column not found to render spending charts.")
                        
                with tab3:
                    st.subheader("🗓️ Detailed Month-Wise Report")
                    
                    amt_col = next((c for c in df.columns if 'amount' in c.lower()), None)
                    date_col = next((c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()), None)
                    
                    if date_col and amt_col:
                        if 'Month' not in df.columns:
                            try:
                                df['Extracted_Month'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%B %Y')
                                month_col_name = 'Extracted_Month'
                            except Exception:
                                month_col_name = date_col
                        else:
                            month_col_name = 'Month'

                        df[amt_col] = pd.to_numeric(df[amt_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                        
                        # Total Monthly Summary
                        monthly_summary = df.groupby(month_col_name)[amt_col].agg(['sum', 'count']).reset_index()
                        monthly_summary.columns = ['Month', 'Total Spent (₹)', 'Total Transactions']
                        monthly_summary = monthly_summary.sort_values(by='Total Spent (₹)', ascending=False)
                        
                        st.markdown("#### 1. Monthly Total Spending Summary")
                        st.dataframe(monthly_summary, use_container_width=True)
                        
                        st.markdown("#### 2. Monthly Spend Trend Chart")
                        st.line_chart(data=monthly_summary, x='Month', y='Total Spent (₹)')
                        
                        # Pivot Breakdown
                        st.markdown("#### 3. Category Breakdown per Month")
                        monthly_cat_pivot = pd.pivot_table(
                            df, 
                            values=amt_col, 
                            index=month_col_name, 
                            columns='Predicted_Category', 
                            aggfunc='sum', 
                            fill_value=0
                        )
                        st.dataframe(monthly_cat_pivot, use_container_width=True)
                    else:
                        st.info("Requires 'Date' and 'Amount' columns for month-wise reporting.")
                        
    except Exception as e:
        st.error(f"Error processing file: {e}")