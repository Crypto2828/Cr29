# crypto.py
import yfinance as yf

def get_stock_data(symbol, period):
    try:
        # استخدم yfinance لجلب البيانات بناءً على رمز العملة والفترة الزمنية
        stock_data = yf.download(symbol, period=period)

        # تحقق مما إذا كانت هناك عمود 'Close' في البيانات
        if 'Close' not in stock_data.columns:
            raise ValueError("بيانات سعر الإغلاق غير متاحة للرمز والفترة المحددين.")

        # إزالة الصفوف التي تحتوي على قيم NaN في عمود 'Close'
        stock_data = stock_data.dropna(subset=['Close'])

        return stock_data
    except Exception as e:
        return {"error": str(e)}
        
