"""Streamlit Cloud 진입점. 실제 대시보드 코드는 web/streamlit_app.py 에 있다.

Streamlit Community Cloud는 메인 파일을 저장소 루트에서 찾으므로, 이 런처가
web/streamlit_app.py 를 그 파일 위치(web/) 기준으로 실행한다.
"""
from pathlib import Path

_app = Path(__file__).resolve().parent / "web" / "streamlit_app.py"
exec(compile(_app.read_text(encoding="utf-8"), str(_app), "exec"),
     {"__name__": "__main__", "__file__": str(_app)})
