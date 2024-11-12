for /L %%i in (1,1,16) do (
    python.exe .\extract_pdf_gemini.py %%i
    timeout 60 /nobreak
)