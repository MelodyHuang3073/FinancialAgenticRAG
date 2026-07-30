import requests

url = "http://localhost:8000/api/upload-file"
sample_csv = "Line Item,2023,2024\nRevenue,1000,1500\nNet Income,300,500\n"

files = {
    'file': ('test_report.csv', sample_csv.encode('utf-8'), 'text/csv')
}
data = {
    'company': '測試科技公司'
}

response = requests.post(url, files=files, data=data)
print("Upload Status Code:", response.status_code)
print("Upload Response:", response.json())

# Query the newly uploaded document
chat_url = "http://localhost:8000/api/chat"
query_payload = {
    "query": "請分析測試科技公司 2023 年至 2024 年 Revenue 成長率 YoY"
}
chat_res = requests.post(chat_url, json=query_payload)
print("\nChat Response:", chat_res.json()["final_answer"])
