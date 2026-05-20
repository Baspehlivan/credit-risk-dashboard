from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="dashboard/app.py",
    path_in_repo="dashboard/app.py",
    repo_id="wiebuch/credit-risk-dashboard",
    repo_type="space",
)
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="wiebuch/credit-risk-dashboard",
    repo_type="space",
)
print("OK - both uploaded")
