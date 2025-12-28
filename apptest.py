import streamlit as st

# ============================================================
# 1. 로그인 상태 확인 함수
# ============================================================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True

    st.title("🔒 매크로 Net liquidity HY Spread")
    
    with st.form("credentials"):
        username = st.text_input("아이디 (ID)", key="username")
        password = st.text_input("비밀번호 (Password)", type="password", key="password")
        submit_btn = st.form_submit_button("로그인", type="primary")

    if submit_btn:
        if username in st.secrets["passwords"] and password == st.secrets["passwords"][username]:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("😕 아이디 또는 비밀번호가 올바르지 않습니다.")
            
    return False

if not check_password():
    st.stop()

# ============================================================
# 메인 임포트
# ============================================================
from fredapi import Fred
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import google.generativeai as genai

warnings.filterwarnings('ignore')

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="퀀트 3콤보 대시보드 + AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Gemini API 설정
# ============================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    GEMINI_ENABLED = True
except Exception as e:
    GEMINI_ENABLED = False
    st.sidebar.warning("⚠️ Gemini API 키가 설정되지 않았습니다. AI 분석 기능이 비활성화됩니다.")

# ============================================================
# AI 분석 함수
# ============================================================
def analyze_with_gemini(analysis_type, data_summary, correlations, signals):
    """
    Gemini API를 사용한 시장 분석 (일반 버전)
    """
    if not GEMINI_ENABLED:
        return "❌ Gemini API가 설정되지 않았습니다. .streamlit/secrets.toml 파일에 GEMINI_API_KEY를 추가하세요."
    
    prompts = {
        "종합분석": f"""
당신은 20년 경력의 거시경제 및 퀀트 투자 전문가입니다.

## 현재 시장 데이터
{data_summary}

## 주요 상관관계
{correlations}

## 현재 시그널
{signals}

다음을 **명확하고 간결하게** 분석해주세요:

1. **현재 거시경제 상황 요약** (3-4문장)
2. **주요 리스크 요인** (2-3가지)
3. **투자 전략 제안** (구체적인 자산별 추천)
4. **주의사항** (1-2가지)

전문가답게, 하지만 일반 투자자도 이해할 수 있게 작성해주세요.
""",
        
        "유동성분석": f"""
당신은 연준(Fed) 정책 및 유동성 전문가입니다.

## Net Liquidity 데이터
{data_summary}

## 상관관계
{correlations}

다음을 분석해주세요:

1. **현재 유동성 상태 평가** (확장/축소/중립)
2. **Fed 정책 방향성** 해석
3. **비트코인/나스닥에 미치는 영향**
4. **향후 3개월 전망**

간결하고 명확하게 답변해주세요.
""",
        
        "달러분석": f"""
당신은 외환 및 글로벌 매크로 전문가입니다.

## Dollar Index vs 위험자산 데이터
{data_summary}

## 상관관계
{correlations}

다음을 분석해주세요:

1. **현재 달러 강도 평가**
2. **달러-비트코인 역상관 상태**
3. **달러-S&P 500 관계**
4. **글로벌 자금 흐름 해석**
5. **투자 전략 제안**

핵심만 간결하게 답변해주세요.
""",
        
        "신용분석": f"""
당신은 신용시장 및 위험관리 전문가입니다.

## High Yield Spread 데이터
{data_summary}

## 상관관계
{correlations}

## 현재 시그널
{signals}

다음을 분석해주세요:

1. **현재 신용시장 상태** (안전/경계/위험)
2. **HY Spread가 의미하는 것**
3. **주식시장/비트코인에 대한 시사점**
4. **리스크 관리 방안**

명확하고 실용적으로 답변해주세요.
""",
        
        "트레이딩전략": f"""
당신은 퀀트 트레이딩 전문가입니다.

## 현재 시그널 종합
{signals}

## 상관관계 매트릭스
{correlations}

## 시장 데이터
{data_summary}

다음을 제시해주세요:

1. **현재 포지션 추천** (매수/매도/관망)
2. **자산별 비중** (BTC/주식/현금)
3. **진입/청산 타이밍**
4. **손절/익절 기준**

구체적이고 실행 가능한 전략을 제시해주세요.
"""
    }
    
    prompt = prompts.get(analysis_type, prompts["종합분석"])
    
    try:
        with st.spinner(f"🤖 Gemini가 {analysis_type} 중..."):
            response = gemini_model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ AI 분석 중 오류 발생: {str(e)}\n\n무료 할당량을 초과했을 수 있습니다. 잠시 후 다시 시도해주세요."

# ============================================================
# AI Deep Dive 분석 함수 (새로 추가)
# ============================================================
def analyze_with_gemini_deep_dive(analysis_type, data_summary, correlations, signals, df_recent, latest):
    """
    Gemini API를 사용한 심층 시장 분석 (Deep Dive)
    """
    if not GEMINI_ENABLED:
        return "❌ Gemini API가 설정되지 않았습니다."
    
    # 추가 통계 정보 생성
    stats_summary = f"""
## 통계 분석 (최근 90일)
- Net Liquidity 변동성: {df_recent['NetLiq'].pct_change().tail(90).std()*100:.2f}%
- BTC 변동성: {df_recent['BTC'].pct_change().tail(90).std()*100:.2f}%
- NASDAQ 변동성: {df_recent['NASDAQ'].pct_change().tail(90).std()*100:.2f}%
- DXY 변동성: {df_recent['DXY'].pct_change().tail(90).std()*100:.2f}%

## 추세 분석
- Net Liquidity 30일 평균: ${df_recent['NetLiq'].tail(30).mean()/1e6:.2f}T
- Net Liquidity 90일 평균: ${df_recent['NetLiq'].tail(90).mean()/1e6:.2f}T
- BTC 30일 평균: ${df_recent['BTC'].tail(30).mean():,.0f}
- BTC 90일 평균: ${df_recent['BTC'].tail(90).mean():,.0f}

## 최근 변화 (7일/30일/90일)
- Net Liquidity: {df_recent['NetLiq'].pct_change(7).iloc[-1]*100:+.2f}% / {df_recent['NetLiq'].pct_change(30).iloc[-1]*100:+.2f}% / {df_recent['NetLiq'].pct_change(90).iloc[-1]*100:+.2f}%
- BTC: {df_recent['BTC'].pct_change(7).iloc[-1]*100:+.2f}% / {df_recent['BTC'].pct_change(30).iloc[-1]*100:+.2f}% / {df_recent['BTC'].pct_change(90).iloc[-1]*100:+.2f}%
- NASDAQ: {df_recent['NASDAQ'].pct_change(7).iloc[-1]*100:+.2f}% / {df_recent['NASDAQ'].pct_change(30).iloc[-1]*100:+.2f}% / {df_recent['NASDAQ'].pct_change(90).iloc[-1]*100:+.2f}%
"""
    
    deep_dive_prompts = {
        "종합분석": f"""
당신은 20년 경력의 거시경제 및 퀀트 투자 전문가입니다. **매우 상세하고 심층적인 분석**을 제공해주세요.

## 현재 시장 데이터
{data_summary}

## 주요 상관관계
{correlations}

## 현재 시그널
{signals}

## 통계 및 추세 분석
{stats_summary}

다음을 **매우 상세하게** 분석해주세요:

### 1. 거시경제 환경 심층 분석 (5-7문장)
- Fed 정책 사이클상 현재 위치
- 유동성 확장/축소의 역사적 맥락
- 주요 중앙은행들의 정책 방향성
- 글로벌 자금 흐름의 변화

### 2. 기술적 분석 및 패턴 인식 (5-6문장)
- 가격 추세와 모멘텀 분석
- 주요 지지/저항 레벨 (데이터 기반)
- 과매수/과매도 구간 판단
- 이동평균선 배열과 시사점

### 3. 리스크 매트릭스 (4가지 이상)
- 단기(1주-1개월) 리스크 요인
- 중기(1-3개월) 리스크 요인
- 구조적 리스크 (장기)
- Black Swan 시나리오

### 4. 시나리오 분석
**Bull Case (낙관적 시나리오 30%):**
- 전개 조건
- 예상 가격 타겟
- 포지셔닝 전략

**Base Case (중립적 시나리오 50%):**
- 전개 조건
- 예상 가격 레인지
- 포지셔닝 전략

**Bear Case (비관적 시나리오 20%):**
- 전개 조건
- 하방 타겟
- 방어 전략

### 5. 구체적 투자 전략 (자산별)
**Bitcoin:**
- 진입 가격대
- 목표가 / 손절가
- 포지션 사이징

**NASDAQ / 주식:**
- 섹터별 전략
- 진입/청산 타이밍
- 리스크 관리

**현금 / 안전자산:**
- 비중 조절 기준
- 재진입 조건

### 6. 향후 3개월 로드맵
- Week 1-2: 단기 전략
- Month 1: 중기 전략
- Month 2-3: 포지션 조정 계획

### 7. 모니터링 체크리스트
- 매일 체크할 지표
- 매주 체크할 지표
- 트리거 이벤트 (포지션 변경 조건)

**전문가답게, 하지만 실행 가능하게 작성해주세요. 수치와 근거를 명확히 제시하세요.**
""",
        
        "유동성분석": f"""
당신은 연준(Fed) 정책 및 유동성 전문가입니다. **심층 유동성 분석**을 제공해주세요.

## Net Liquidity 데이터
{data_summary}

## 상관관계
{correlations}

## 통계 분석
{stats_summary}

다음을 **매우 상세하게** 분석해주세요:

### 1. Fed 대차대조표 심층 분석 (5-6문장)
- WALCL (Fed 총자산) 추세와 의미
- TGA (재무부 계좌) 변화와 정책 시사점
- RRP (역RP) 수준과 은행 유동성 상태
- Net Liquidity의 역사적 위치

### 2. 유동성 사이클 분석
- 현재 사이클상 위치 (확장/정점/축소/저점)
- 과거 유사 패턴과 비교
- 전환점 시그널 (Leading Indicators)

### 3. 시장 영향 메커니즘
- Net Liquidity → Bitcoin 전달 경로
- Net Liquidity → 주식시장 영향 시차
- 유동성 변화의 선행/후행 지표

### 4. Fed 정책 전망 (3-6개월)
- FOMC 회의 일정과 예상 시나리오
- QT (양적긴축) 지속 여부
- 정책 전환 가능성과 조건

### 5. 투자 전략 (유동성 기반)
**확장 구간 전략:**
- 공격적 포지셔닝 타이밍
- 레버리지 활용 방안

**축소 구간 전략:**
- 방어적 포지셔닝
- 현금 비중 확대 기준

**전환점 대응:**
- 조기 시그널 포착 방법
- 포지션 전환 타이밍

### 6. 리스크 시나리오
- 급격한 유동성 축소 시나리오
- 정책 오류 가능성
- 비상 대응 계획

**수치와 역사적 데이터를 활용하여 설득력 있게 작성해주세요.**
""",
        
        "달러분석": f"""
당신은 외환 및 글로벌 매크로 전문가입니다. **심층 달러 분석**을 제공해주세요.

## Dollar Index vs 위험자산 데이터
{data_summary}

## 상관관계
{correlations}

## 통계 분석
{stats_summary}

다음을 **매우 상세하게** 분석해주세요:

### 1. 달러 강도 심층 분석 (5-6문장)
- DXY 현재 수준의 역사적 의미
- 주요 통화 (EUR, JPY, GBP) 대비 달러 강도
- 금리 차이와 달러 움직임
- 실질 달러 vs 명목 달러

### 2. 달러-비트코인 역학 분석
- 역상관 메커니즘 설명
- 현재 상관계수의 의미
- 역상관 붕괴 시나리오
- 과거 패턴과 비교

### 3. 달러-S&P 500 관계 분석
- 달러 강세가 미국 주식에 미치는 영향
- 수출 기업 vs 내수 기업 영향 차이
- 달러-주식 상관관계 변화 추이

### 4. 글로벌 자금 흐름
- 신흥국 → 선진국 흐름
- 안전자산 선호도 (Risk-off 정도)
- 캐리 트레이드 상황
- 달러 환류 vs 유출

### 5. 지정학적 요인
- 미중 관계와 달러
- 에너지 가격과 달러 연계성
- BRICS 탈달러화 영향

### 6. 시나리오별 전략
**달러 강세 시나리오:**
- BTC/알트코인 대응
- 신흥국 자산 전략
- 방어 포트폴리오

**달러 약세 시나리오:**
- 위험자산 공격적 배분
- 상품/귀금속 전략
- 레버리지 활용

### 7. 트레이딩 전략
- DXY 기준 매매 시그널
- 옵션 전략 (달러 헤지)
- 포트폴리오 통화 배분

**글로벌 매크로 관점에서 종합적으로 분석해주세요.**
""",
        
        "신용분석": f"""
당신은 신용시장 및 위험관리 전문가입니다. **심층 신용 분석**을 제공해주세요.

## High Yield Spread 데이터
{data_summary}

## 현재 시그널
{signals}

## 상관관계
{correlations}

## 통계 분석
{stats_summary}

다음을 **매우 상세하게** 분석해주세요:

### 1. HY Spread 심층 해석 (5-6문장)
- 현재 스프레드의 역사적 위치
- Investment Grade vs High Yield 스프레드 비교
- 크레딧 사이클상 위치
- 디폴트율 전망

### 2. 신용 리스크 분석
- 기업 부채 수준과 지속가능성
- 이자 커버리지 비율 추세
- 리파이낸싱 리스크 (만기 wall)
- 섹터별 신용 건전성

### 3. HY Spread의 선행성 분석
- 주식 시장 대비 선행/후행
- 비트코인 시장과의 관계
- 과거 경기침체 전 패턴
- 현재와 과거 비교
- False Signal vs True Signal 구분

### 4. Divergence 심층 분석
- S&P 상승 + HY Spread 상승의 의미
- 과거 Divergence 사례 연구
- 해소 패턴 (수렴 방향 예측)
- 지속 기간과 거래 전략

### 5. 시나리오별 대응
**신용경색 시나리오 (HY Spread 급등):**
- 조기 경보 시그널
- 포트폴리오 방어 전략
- 현금 확보 계획

**정상화 시나리오 (스프레드 안정):**
- 리스크 재진입 타이밍
- 섹터/종목 선별 전략

### 6. 리스크 관리 프레임워크
- Stop-loss 기준 (HY Spread 기준)
- 포지션 사이징 공식
- 헤지 전략 (옵션, 인버스 ETF)

### 7. 모니터링 체크리스트
- 일일 체크: HY Spread, 주식 가격, BTC
- 주간 체크: 신용 등급 변화, 디폴트
- 월간 체크: 기업 실적, 부채 추이

**신용시장 전문가 관점에서 리스크를 정량화하여 제시해주세요.**
""",
        
        "트레이딩전략": f"""
당신은 퀀트 트레이딩 전문가입니다. **실행 가능한 상세 트레이딩 전략**을 제공해주세요.

## 현재 시그널 종합
{signals}

## 상관관계 매트릭스
{correlations}

## 시장 데이터
{data_summary}

## 통계 분석
{stats_summary}

다음을 **매우 구체적으로** 제시해주세요:

### 1. 현재 시장 진단
- 시장 regime (Trending/Mean-reverting/Volatile)
- Risk-on vs Risk-off 정도 (0-100 점수)
- 과매수/과매도 지표

### 2. 포트폴리오 구성 (구체적 비중)
**현재 추천 배분:**
- Bitcoin: ___%
- 주식 (NASDAQ): ___%
- 현금/안전자산: ___%
- 이유와 근거

**리밸런싱 조건:**
- 언제 비중을 조정할 것인가
- 트리거 가격/지표

### 3. 진입 전략 (자산별)
**Bitcoin 진입:**
- 1차 진입가: $____
- 2차 진입가: $____  
- 평균 단가 전략
- 포지션 사이즈: 총 자산의 ___%

**주식 진입:**
- NASDAQ 레벨: ____포인트
- 분할 매수 계획
- 섹터 배분

### 4. 청산 전략
**익절 기준:**
- 1차 익절: +___% (물량 ___%)
- 2차 익절: +___% (물량 ___%)
- 최종 익절: +___% (잔량 전부)

**손절 기준:**
- 손절 라인: -___%
- 타임 스탑 (시간 기반): ___일
- 시그널 전환 시 즉시 청산 조건

### 5. 리스크 관리
- 1회 거래 최대 리스크: ___%
- 총 포트폴리오 리스크: ___%
- 최대 드로다운 허용: ___%
- 연속 손실 시 대응 (___회 연속 손실 시 휴식)

### 6. 시나리오별 대응 플레이북
**시나리오 A: Net Liquidity 급격 확장**
→ 액션: ________________
→ 타겟: ________________

**시나리오 B: HY Spread 5% 돌파**
→ 액션: ________________
→ 타겟: ________________

**시나리오 C: DXY 급등**
→ 액션: ________________
→ 타겟: ________________

### 7. 일간/주간 체크리스트
**매일 체크할 것:**
- [ ] Net Liquidity 확인
- [ ] DXY 레벨 확인
- [ ] HY Spread 확인
- [ ] 포지션 손익 계산

**매주 체크할 것:**
- [ ] 상관관계 변화
- [ ] 포트폴리오 리밸런싱 필요성
- [ ] 다음 주 주요 이벤트

### 8. 백테스트 아이디어
- 과거 유사 상황에서의 성과
- Win rate / Profit factor 추정
- 최대 드로다운 예상

**실전에서 바로 실행 가능하도록, 수치와 조건을 명확히 제시해주세요. 
애매한 표현 없이, 구체적인 가격과 %로 제시하세요.**
"""
    }
    
    prompt = deep_dive_prompts.get(analysis_type, deep_dive_prompts["종합분석"])
    
    try:
        with st.spinner(f"🔬 Gemini가 Deep Dive {analysis_type} 중... (시간이 조금 걸릴 수 있습니다)"):
            response = gemini_model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ AI Deep Dive 분석 중 오류 발생: {str(e)}\n\n무료 할당량을 초과했을 수 있습니다. 잠시 후 다시 시도해주세요."

def get_data_summary(df_recent, latest, netliq_60d):
    """현재 데이터 요약 생성"""
    return f"""
- Net Liquidity: ${latest['NetLiq']/1e6:.2f}T ({netliq_60d:+.2f}% / 60일)
- Bitcoin: ${latest['BTC']:,.0f} ({df_recent['BTC'].pct_change(30).iloc[-1]*100:+.2f}% / 30일)
- NASDAQ: {latest['NASDAQ']:,.0f} ({df_recent['NASDAQ'].pct_change(30).iloc[-1]*100:+.2f}% / 30일)
- S&P 500: {latest['SP500']:,.0f} ({df_recent['SP500'].pct_change(30).iloc[-1]*100:+.2f}% / 30일)
- Dollar Index: {latest['DXY']:.2f} ({df_recent['DXY'].pct_change(30).iloc[-1]*100:+.2f}% / 30일)
- HY Spread: {latest['HYSpread']:.2f}%
"""

def get_correlations_summary(corr_matrix):
    """상관관계 요약 생성"""
    return f"""
- Net Liquidity ↔ BTC: {corr_matrix.loc['NetLiq', 'BTC']:.3f}
- Net Liquidity ↔ NASDAQ: {corr_matrix.loc['NetLiq', 'NASDAQ']:.3f}
- Dollar Index ↔ BTC: {corr_matrix.loc['DXY', 'BTC']:.3f}
- Dollar Index ↔ S&P500: {corr_matrix.loc['DXY', 'SP500']:.3f}
- HY Spread ↔ S&P500: {corr_matrix.loc['HYSpread', 'SP500']:.3f}
- HY Spread ↔ BTC: {corr_matrix.loc['HYSpread', 'BTC']:.3f}
"""

def get_signals_summary(netliq_60d, latest, corr_dxy_btc, recent_divergence):
    """시그널 요약 생성"""
    netliq_signal = "확장 🟢" if netliq_60d > 2 else ("축소 🔴" if netliq_60d < -2 else "중립 ⚪")
    dxy_signal = "강한 역상관 🟢" if corr_dxy_btc < -0.5 else ("비정상 동행 🔴" if corr_dxy_btc > 0 else "약한 역상관 ⚪")
    hy_signal = "위험 🔴" if latest['HYSpread'] > 5.0 else ("경계 🟡" if latest['HYSpread'] > 4.0 else "안정 🟢")
    div_signal = f"발생 ({recent_divergence}일) 🔴" if recent_divergence > 0 else "없음 🟢"
    
    return f"""
1. Net Liquidity: {netliq_signal} ({netliq_60d:+.2f}%)
2. DXY-BTC 관계: {dxy_signal} ({corr_dxy_btc:.3f})
3. HY Spread: {hy_signal} ({latest['HYSpread']:.2f}%)
4. Divergence: {div_signal}
"""

# ============================================================
# 사이드바 설정
# ============================================================
st.sidebar.title("⚙️ 분석 설정")
st.sidebar.markdown("---")

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception as e:
    st.error("⚠️ FRED API 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

period_options = {
    "최근 1년": 365,
    "최근 2년": 365*2,
    "최근 3년": 365*3,
    "최근 5년": 365*5
}
selected_period = st.sidebar.selectbox(
    "📅 분석 기간",
    list(period_options.keys()),
    index=2
)
days = period_options[selected_period]

window = st.sidebar.slider(
    "📈 상관계수 롤링 윈도우 (일)",
    min_value=30,
    max_value=180,
    value=90,
    step=10
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI 분석 상태")
if GEMINI_ENABLED:
    st.sidebar.success("✅ Gemini AI 활성화")
    st.sidebar.info("""
    **무료 할당량:**
    - 분당 15 요청
    - 일일 1,500 요청
    """)
else:
    st.sidebar.error("❌ Gemini AI 비활성화")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 대시보드 정보")
st.sidebar.info("""
**분석 지표:**
- Net Liquidity (Fed 유동성)
- Dollar Index (달러 강도)
- HY Spread (신용 스프레드)
- Bitcoin, NASDAQ, S&P 500

**데이터 출처:** FRED API
**AI 엔진:** Google Gemini 2.0 Flash
""")

# ============================================================
# 메인 타이틀
# ============================================================
st.title("매크로 Net liquidity HY Spread")
st.markdown("""
**Fed 유동성, 달러 인덱스, HY Spread **  
실시간 FRED 데이터 기반 인터랙티브 분석 
""")
st.markdown("---")

# ============================================================
# 데이터 로딩 함수
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(api_key, days):
    """FRED API에서 데이터 로드"""
    try:
        fred = Fred(api_key=api_key)
        start_date = datetime.now() - timedelta(days=days)
        
        walcl = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date)
        dxy = fred.get_series('DTWEXAFEGS', observation_start=start_date)
        hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date)
        btc = fred.get_series('CBBTCUSD', observation_start=start_date)
        nasdaq = fred.get_series('NASDAQCOM', observation_start=start_date)
        sp500 = fred.get_series('SP500', observation_start=start_date)
        
        return {
            'walcl': walcl, 'tga': tga, 'rrp': rrp,
            'dxy': dxy, 'hy_spread': hy_spread,
            'btc': btc, 'nasdaq': nasdaq, 'sp500': sp500
        }
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {str(e)}")
        return None

def process_data(raw_data):
    """Net Liquidity 계산 및 데이터 통합"""
    try:
        df_liq = pd.DataFrame({
            'WALCL_Mn': raw_data['walcl'],
            'TGA_Mn': raw_data['tga'],
            'RRP_Bn': raw_data['rrp']
        })
        
        df_liq['RRP_Mn'] = df_liq['RRP_Bn'] * 1000
        df_liq = df_liq.fillna(method='ffill').dropna()
        df_liq['NetLiquidity'] = (
            df_liq['WALCL_Mn'] - df_liq['TGA_Mn'] - df_liq['RRP_Mn']
        )
        
        df_all = pd.DataFrame({
            'NetLiq': df_liq['NetLiquidity'],
            'DXY': raw_data['dxy'],
            'HYSpread': raw_data['hy_spread'],
            'BTC': raw_data['btc'],
            'NASDAQ': raw_data['nasdaq'],
            'SP500': raw_data['sp500']
        })
        
        df_all = df_all.fillna(method='ffill').dropna()
        return df_all
        
    except Exception as e:
        st.error(f"❌ 데이터 처리 실패: {str(e)}")
        return None

def zscore(series):
    """Z-score 정규화"""
    return (series - series.mean()) / series.std()

# ============================================================
# 데이터 로드
# ============================================================
with st.spinner("🔄 FRED 데이터 다운로드 중..."):
    raw_data = load_data(FRED_API_KEY, days)

if raw_data is None:
    st.stop()

df_recent = process_data(raw_data)

if df_recent is None:
    st.stop()

st.success(f"✅ 데이터 로드 완료: {df_recent.index[0].date()} ~ {df_recent.index[-1].date()} ({len(df_recent)}개 포인트)")

# ============================================================
# 최신 지표 요약
# ============================================================
latest = df_recent.iloc[-1]
netliq_60d = df_recent['NetLiq'].pct_change(periods=60).iloc[-1] * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Net Liquidity",
        f"${latest['NetLiq']/1e6:.2f}T",
        f"{netliq_60d:+.2f}% (60일)"
    )

with col2:
    btc_change = df_recent['BTC'].pct_change(periods=30).iloc[-1] * 100
    st.metric(
        "₿ Bitcoin",
        f"${latest['BTC']:,.0f}",
        f"{btc_change:+.2f}% (30일)"
    )

with col3:
    dxy_change = df_recent['DXY'].pct_change(periods=30).iloc[-1] * 100
    st.metric(
        "💵 Dollar Index",
        f"{latest['DXY']:.2f}",
        f"{dxy_change:+.2f}% (30일)"
    )

with col4:
    hy_status = "🚨 위험" if latest['HYSpread'] > 5 else "✅ 정상"
    st.metric(
        "⚠️ HY Spread",
        f"{latest['HYSpread']:.2f}%",
        hy_status
    )

st.markdown("---")

# ============================================================
# 상관계수 계산 (전역 변수로 사용)
# ============================================================
ret = df_recent[['NetLiq', 'BTC', 'NASDAQ', 'DXY', 'HYSpread', 'SP500']].pct_change().dropna()
corr_btc = ret['NetLiq'].rolling(window).corr(ret['BTC'])
corr_nasdaq = ret['NetLiq'].rolling(window).corr(ret['NASDAQ'])
corr_dxy_btc = ret['DXY'].rolling(window).corr(ret['BTC'])
corr_dxy_sp = ret['DXY'].rolling(window).corr(ret['SP500'])  # 추가
corr_hy_sp = ret['HYSpread'].rolling(window).corr(ret['SP500'])
corr_hy_btc = ret['HYSpread'].rolling(window).corr(ret['BTC'])  # 추가
corr_matrix = df_recent[['NetLiq', 'DXY', 'HYSpread', 'BTC', 'NASDAQ', 'SP500']].corr()

# Divergence 계산
sp_ret = df_recent['SP500'].pct_change(periods=20)
hy_change = df_recent['HYSpread'].diff(periods=20)
divergence = (sp_ret > 0) & (hy_change > 0)
recent_divergence = divergence.tail(5).sum()

# ============================================================
# 탭 구성
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 콤보 1: Net Liquidity",
    "💵 콤보 2: Dollar Index",
    "⚠️ 콤보 3: HY Spread",
    "🎯 종합 대시보드",
    "📊 트레이딩 시그널",
    "🤖 AI 분석"
])

# ============================================================
# TAB 1: Net Liquidity 분석
# ============================================================
with tab1:
    st.header("📈 콤보 1: Net Liquidity 분석")
    st.markdown("**Fed 총자산 - 재무부 계좌 - 역RP = Net Liquidity**")
    
    df_z1 = df_recent[['NetLiq', 'BTC', 'NASDAQ']].apply(zscore)
    netliq_change = df_recent['NetLiq'].pct_change(periods=60) * 100
    
    fig1 = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Net Liquidity vs BTC/NASDAQ (Z-score)',
            f'Net Liquidity 상관계수 ({window}일 롤링)',
            'Net Liquidity 60일 변화율 (유동성 확장/축소)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.3, 0.35]
    )
    
    fig1.add_trace(
        go.Scatter(x=df_z1.index, y=df_z1['NetLiq'],
                   name='Net Liquidity', line=dict(color='#2E86AB', width=2.5)),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=df_z1.index, y=df_z1['BTC'],
                   name='Bitcoin', line=dict(color='#F77F00', width=2.5)),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=df_z1.index, y=df_z1['NASDAQ'],
                   name='NASDAQ', line=dict(color='#06A77D', width=2.5)),
        row=1, col=1
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    fig1.add_trace(
        go.Scatter(x=corr_btc.index, y=corr_btc,
                   name='Corr(NetLiq, BTC)',
                   line=dict(color='#F77F00', width=2.5),
                   fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.2)'),
        row=2, col=1
    )
    fig1.add_trace(
        go.Scatter(x=corr_nasdaq.index, y=corr_nasdaq,
                   name='Corr(NetLiq, NASDAQ)',
                   line=dict(color='#06A77D', width=2.5),
                   fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.2)'),
        row=2, col=1
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    expansion = netliq_change[netliq_change > 0]
    fig1.add_trace(
        go.Scatter(x=expansion.index, y=expansion,
                   name='확장 구간 🟢',
                   line=dict(color='#06A77D', width=0),
                   fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.4)'),
        row=3, col=1
    )
    
    contraction = netliq_change[netliq_change <= 0]
    fig1.add_trace(
        go.Scatter(x=contraction.index, y=contraction,
                   name='축소 구간 🔴',
                   line=dict(color='#D62828', width=0),
                   fill='tozeroy', fillcolor='rgba(214, 40, 40, 0.4)'),
        row=3, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=netliq_change.index, y=netliq_change,
                   name='변화율', line=dict(color='black', width=2),
                   showlegend=False),
        row=3, col=1
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    fig1.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig1.update_yaxes(title_text="Z-score", row=1, col=1)
    fig1.update_yaxes(title_text="Correlation", row=2, col=1)
    fig1.update_yaxes(title_text="변화율 (%)", row=3, col=1)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("### 📌 분석 인사이트")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **최근 상관계수**
        - NetLiq ↔ BTC: {corr_btc.iloc[-1]:.3f}
        - NetLiq ↔ NASDAQ: {corr_nasdaq.iloc[-1]:.3f}
        """)
    with col2:
        signal = "🟢 확장 (리스크 온)" if netliq_60d > 0 else "🔴 축소 (리스크 오프)"
        st.warning(f"""
        **현재 유동성 상태**
        - 60일 변화: {netliq_60d:+.2f}%
        - 시그널: {signal}
        """)

# ============================================================
# TAB 2: Dollar Index vs BTC & S&P 500 (업데이트)
# ============================================================
with tab2:
    st.header("💵 콤보 2: Dollar Index 분석")
    st.markdown("**달러 강세와 위험자산(BTC, S&P 500)의 관계**")
    
    df_z2 = pd.DataFrame({
        'DXY_Inverted': zscore(-df_recent['DXY']),
        'BTC': zscore(df_recent['BTC']),
        'SP500': zscore(df_recent['SP500'])
    })
    
    fig2 = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Dollar Index (반전) vs BTC/S&P 500 (Z-score)',
            f'Dollar Index 상관계수 ({window}일 롤링)',
            'Dollar Index 원본 차트'
        ),
        vertical_spacing=0.10,
        row_heights=[0.35, 0.35, 0.30]
    )
    
    # 첫 번째 차트: DXY 반전 vs BTC & S&P 500
    fig2.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['DXY_Inverted'],
                   name='Dollar Index (반전)',
                   line=dict(color='#D62828', width=2.5)),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['BTC'],
                   name='Bitcoin',
                   line=dict(color='#F77F00', width=2.5)),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['SP500'],
                   name='S&P 500',
                   line=dict(color='#2E86AB', width=2.5)),
        row=1, col=1
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 두 번째 차트: 상관계수
    fig2.add_trace(
        go.Scatter(x=corr_dxy_btc.index, y=corr_dxy_btc,
                   name='Corr(DXY, BTC)',
                   line=dict(color='#F77F00', width=2.5),
                   fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.3)'),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(x=corr_dxy_sp.index, y=corr_dxy_sp,
                   name='Corr(DXY, S&P500)',
                   line=dict(color='#2E86AB', width=2.5),
                   fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.3)'),
        row=2, col=1
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig2.add_hline(y=-0.5, line_dash="dot", line_color="green", opacity=0.7, 
                   annotation_text="강한 역상관", row=2, col=1)
    
    # 세 번째 차트: DXY 원본
    fig2.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['DXY'],
                   name='Dollar Index',
                   line=dict(color='#D62828', width=2.5),
                   fill='tozeroy', fillcolor='rgba(214, 40, 40, 0.2)'),
        row=3, col=1
    )
    
    fig2.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig2.update_yaxes(title_text="Z-score", row=1, col=1)
    fig2.update_yaxes(title_text="Correlation", row=2, col=1)
    fig2.update_yaxes(title_text="Dollar Index", row=3, col=1)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 📌 분석 인사이트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔶 DXY vs Bitcoin")
        if corr_dxy_btc.iloc[-1] < -0.5:
            st.success(f"""
            ✅ **강한 역상관 감지** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
            - 달러 약세 시 비트코인 강세 예상
            - DXY 하락 구간에서 BTC 매수 기회
            """)
        elif corr_dxy_btc.iloc[-1] > 0:
            st.error(f"""
            ⚠️ **비정상 동행** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
            - 달러와 비트코인이 같은 방향으로 움직임
            - 리스크 오프 모드 가능성
            """)
        else:
            st.info(f"""
            ⏸️ **역상관 약화** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
            - 달러와 비트코인의 연관성 감소
            - 다른 요인이 가격에 더 큰 영향
            """)
    
    with col2:
        st.markdown("#### 🔷 DXY vs S&P 500")
        if corr_dxy_sp.iloc[-1] < -0.3:
            st.success(f"""
            ✅ **역상관 관계** (상관계수: {corr_dxy_sp.iloc[-1]:.3f})
            - 달러 약세 = 주식 강세
            - 수출 기업에 유리한 환경
            """)
        elif corr_dxy_sp.iloc[-1] > 0.3:
            st.warning(f"""
            ⚠️ **양의 상관** (상관계수: {corr_dxy_sp.iloc[-1]:.3f})
            - 달러 강세 속 주식도 상승
            - 미국 경제 독주 가능성
            """)
        else:
            st.info(f"""
            ⏸️ **약한 상관관계** (상관계수: {corr_dxy_sp.iloc[-1]:.3f})
            - 달러와 주식의 독립적 움직임
            - 개별 요인 우선 작용
            """)
    
    st.markdown("---")
    st.markdown("### 💡 달러 강도 해석")
    
    dxy_current = latest['DXY']
    dxy_ma30 = df_recent['DXY'].tail(30).mean()
    dxy_ma90 = df_recent['DXY'].tail(90).mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("현재 DXY", f"{dxy_current:.2f}")
    with col2:
        st.metric("30일 평균", f"{dxy_ma30:.2f}", f"{((dxy_current - dxy_ma30) / dxy_ma30 * 100):+.2f}%")
    with col3:
        st.metric("90일 평균", f"{dxy_ma90:.2f}", f"{((dxy_current - dxy_ma90) / dxy_ma90 * 100):+.2f}%")
    
    if dxy_current > dxy_ma30 and dxy_current > dxy_ma90:
        st.warning("""
        📈 **달러 강세 구간**
        - 위험자산 역풍 가능성
        - 신흥국 통화 약세 압력
        - 글로벌 유동성 긴축 효과
        """)
    elif dxy_current < dxy_ma30 and dxy_current < dxy_ma90:
        st.success("""
        📉 **달러 약세 구간**
        - 위험자산 순풍
        - 상품/원자재 가격 상승 가능
        - 글로벌 유동성 확장 효과
        """)
    else:
        st.info("""
        ⚖️ **달러 중립 구간**
        - 방향성 불명확
        - 다른 지표 중시 필요
        """)

# ============================================================
# TAB 3: HY Spread 분석 (업데이트)
# ============================================================
with tab3:
    st.header("⚠️ 콤보 3: High Yield Spread 분석")
    st.markdown("**HY Spread 상승 = 신용 위험 증가 = 위험자산 경계**")
    
    # Z-score 정규화 추가
    df_z3 = pd.DataFrame({
        'HYSpread': zscore(df_recent['HYSpread']),
        'SP500': zscore(df_recent['SP500']),
        'BTC': zscore(df_recent['BTC'])
    })
    
    fig3 = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'High Yield Spread vs S&P 500 / BTC (Z-score)',
            f'HY Spread 상관계수 ({window}일 롤링)',
            'Divergence 감지: S&P 상승 + HY Spread 상승 (매도 신호)',
            'HY Spread 원본 차트'
        ),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.25, 0.20]
    )
    
    # 첫 번째 차트: HY Spread vs S&P 500 & BTC (Z-score)
    fig3.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['HYSpread'],
                   name='HY Spread',
                   line=dict(color='#D62828', width=2.5)),
        row=1, col=1
    )
    fig3.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['SP500'],
                   name='S&P 500',
                   line=dict(color='#2E86AB', width=2.5)),
        row=1, col=1
    )
    fig3.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['BTC'],
                   name='Bitcoin',
                   line=dict(color='#F77F00', width=2)),
        row=1, col=1
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 두 번째 차트: 상관계수
    fig3.add_trace(
        go.Scatter(x=corr_hy_sp.index, y=corr_hy_sp,
                   name='Corr(HY, S&P500)',
                   line=dict(color='#2E86AB', width=2.5),
                   fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.3)'),
        row=2, col=1
    )
    fig3.add_trace(
        go.Scatter(x=corr_hy_btc.index, y=corr_hy_btc,
                   name='Corr(HY, BTC)',
                   line=dict(color='#F77F00', width=2.5),
                   fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.3)'),
        row=2, col=1
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 세 번째 차트: Divergence (원본 가격 유지)
    fig3.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['SP500'],
                   name='S&P 500',
                   line=dict(color='#2E86AB', width=2), opacity=0.6),
        row=3, col=1
    )
    fig3.add_trace(
        go.Scatter(x=df_recent[divergence].index,
                   y=df_recent.loc[divergence, 'SP500'],
                   name='Divergence 경고 ⚠️',
                   mode='markers',
                   marker=dict(color='red', size=10, symbol='diamond')),
        row=3, col=1
    )
    
    # 네 번째 차트: HY Spread 원본
    fig3.add_trace(
        go.Scatter(x=df_recent.index, y=df_recent['HYSpread'],
                   name='HY Spread',
                   line=dict(color='#D62828', width=2.5),
                   fill='tozeroy', fillcolor='rgba(214, 40, 40, 0.2)'),
        row=4, col=1
    )
    fig3.add_hline(y=4.0, line_dash="dot", line_color="orange", opacity=0.7,
                   annotation_text="경계 (4%)", row=4, col=1)
    fig3.add_hline(y=5.0, line_dash="dash", line_color="darkred", opacity=0.8,
                   annotation_text="위험 (5%)", row=4, col=1)
    
    fig3.update_layout(
        height=1400,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig3.update_yaxes(title_text="Z-score", row=1, col=1)
    fig3.update_yaxes(title_text="Correlation", row=2, col=1)
    fig3.update_yaxes(title_text="S&P 500", row=3, col=1)
    fig3.update_yaxes(title_text="HY Spread (%)", row=4, col=1)
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("### 📌 분석 인사이트")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ⚠️ HY Spread 상태")
        if latest['HYSpread'] > 5.0:
            st.error(f"""
            🚨 **위기 임계점 초과**
            - 현재: {latest['HYSpread']:.2f}%
            - 신용 시장 경색
            - 주식/BTC 하락 위험
            """)
        elif latest['HYSpread'] > 4.0:
            st.warning(f"""
            ⚠️ **경계 구간**
            - 현재: {latest['HYSpread']:.2f}%
            - 주의 필요
            - 포지션 축소 고려
            """)
        else:
            st.success(f"""
            ✅ **정상 구간**
            - 현재: {latest['HYSpread']:.2f}%
            - 신용 시장 안정
            """)
    
    with col2:
        st.markdown("#### 📊 상관관계 분석")
        st.info(f"""
        **HY Spread와 위험자산**
        - vs S&P 500: {corr_hy_sp.iloc[-1]:.3f}
        - vs Bitcoin: {corr_hy_btc.iloc[-1]:.3f}
        
        {'음의 상관: HY↑ = 자산↓' if corr_hy_sp.iloc[-1] < 0 else '양의 상관: 비정상'}
        """)
    
    with col3:
        st.markdown("#### 🔴 Divergence")
        if recent_divergence > 0:
            st.error(f"""
            ⚠️ **경고 발생**
            - 최근 5일 중 {recent_divergence}일
            - S&P↑ + HY Spread↑
            - 허위 랠리 가능성
            """)
        else:
            st.success("✅ 최근 없음")
    
    st.markdown("---")
    st.markdown("### 💡 신용 시장 해석")
    
    hy_ma30 = df_recent['HYSpread'].tail(30).mean()
    hy_ma90 = df_recent['HYSpread'].tail(90).mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("현재 HY Spread", f"{latest['HYSpread']:.2f}%")
    with col2:
        st.metric("30일 평균", f"{hy_ma30:.2f}%", f"{(latest['HYSpread'] - hy_ma30):+.2f}%p")
    with col3:
        st.metric("90일 평균", f"{hy_ma90:.2f}%", f"{(latest['HYSpread'] - hy_ma90):+.2f}%p")
    
    if latest['HYSpread'] > hy_ma30 and latest['HYSpread'] > hy_ma90:
        st.warning("""
        📈 **HY Spread 상승 추세**
        - 기업 신용 위험 증가
        - 경기 둔화 신호 가능
        - 리스크 자산 방어적 접근
        """)
    elif latest['HYSpread'] < hy_ma30 and latest['HYSpread'] < hy_ma90:
        st.success("""
        📉 **HY Spread 하락 추세**
        - 신용 시장 개선
        - 리스크 선호 증가
        - 공격적 포지셔닝 가능
        """)
    else:
        st.info("""
        ⚖️ **HY Spread 중립**
        - 신용 시장 안정
        - 다른 지표 참고
        """)

# ============================================================
# TAB 4: 종합 대시보드 (업데이트)
# ============================================================
with tab4:
    st.header("🎯 종합 대시보드")
    
    fig_dashboard = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Net Liquidity + BTC/NASDAQ (Z-score)',
            '상관계수 히트맵',
            'Dollar Index (반전) vs BTC/S&P500 (Z-score)',
            'DXY 상관계수',
            'HY Spread vs S&P500/BTC (Z-score)',
            'HY Spread 상관계수'
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Row 1, Col 1: Net Liquidity
    df_z_all = df_recent[['NetLiq', 'BTC', 'NASDAQ']].apply(zscore)
    fig_dashboard.add_trace(
        go.Scatter(x=df_z_all.index, y=df_z_all['NetLiq'],
                   name='Net Liquidity', line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z_all.index, y=df_z_all['BTC'],
                   name='Bitcoin', line=dict(color='#F77F00', width=2)),
        row=1, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z_all.index, y=df_z_all['NASDAQ'],
                   name='NASDAQ', line=dict(color='#06A77D', width=2)),
        row=1, col=1
    )
    fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Row 1, Col 2: 상관계수 히트맵
    fig_dashboard.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ),
        row=1, col=2
    )
    
    # Row 2, Col 1: DXY vs BTC/S&P500
    fig_dashboard.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['DXY_Inverted'],
                   name='DXY (반전)', line=dict(color='#D62828', width=2)),
        row=2, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['BTC'],
                   name='BTC', line=dict(color='#F77F00', width=2)),
        row=2, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z2.index, y=df_z2['SP500'],
                   name='S&P500', line=dict(color='#2E86AB', width=2)),
        row=2, col=1
    )
    fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Row 2, Col 2: DXY 상관계수
    fig_dashboard.add_trace(
        go.Scatter(x=corr_dxy_btc.index, y=corr_dxy_btc,
                   name='Corr(DXY, BTC)', line=dict(color='#F77F00', width=2),
                   fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.2)'),
        row=2, col=2
    )
    fig_dashboard.add_trace(
        go.Scatter(x=corr_dxy_sp.index, y=corr_dxy_sp,
                   name='Corr(DXY, S&P500)', line=dict(color='#2E86AB', width=2),
                   fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.2)'),
        row=2, col=2
    )
    fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)
    
    # Row 3, Col 1: HY Spread vs S&P500/BTC (Z-score로 변경)
    fig_dashboard.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['HYSpread'],
                   name='HY Spread', line=dict(color='#D62828', width=2.5)),
        row=3, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['SP500'],
                   name='S&P 500', line=dict(color='#2E86AB', width=2)),
        row=3, col=1
    )
    fig_dashboard.add_trace(
        go.Scatter(x=df_z3.index, y=df_z3['BTC'],
                   name='Bitcoin', line=dict(color='#F77F00', width=1.5), opacity=0.7),
        row=3, col=1
    )
    fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Row 3, Col 2: HY Spread 상관계수
    fig_dashboard.add_trace(
        go.Scatter(x=corr_hy_sp.index, y=corr_hy_sp,
                   name='Corr(HY, S&P500)', line=dict(color='#2E86AB', width=2),
                   fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.2)'),
        row=3, col=2
    )
    fig_dashboard.add_trace(
        go.Scatter(x=corr_hy_btc.index, y=corr_hy_btc,
                   name='Corr(HY, BTC)', line=dict(color='#F77F00', width=2),
                   fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.2)'),
        row=3, col=2
    )
    fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=2)
    
    fig_dashboard.update_layout(
        height=1400,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig_dashboard.update_yaxes(title_text="Z-score", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="Z-score", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Correlation", row=2, col=2)
    fig_dashboard.update_yaxes(title_text="Z-score", row=3, col=1)  # 변경됨!
    fig_dashboard.update_yaxes(title_text="Correlation", row=3, col=2)
    
    st.plotly_chart(fig_dashboard, use_container_width=True)
    
    st.markdown("### 📊 상관계수 매트릭스 (상세)")
    st.dataframe(corr_matrix.round(3), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 주요 상관관계 요약")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Net Liquidity")
        st.info(f"""
        - vs BTC: {corr_matrix.loc['NetLiq', 'BTC']:.3f}
        - vs NASDAQ: {corr_matrix.loc['NetLiq', 'NASDAQ']:.3f}
        - vs S&P500: {corr_matrix.loc['NetLiq', 'SP500']:.3f}
        """)
    
    with col2:
        st.markdown("#### 💵 Dollar Index")
        st.info(f"""
        - vs BTC: {corr_matrix.loc['DXY', 'BTC']:.3f}
        - vs NASDAQ: {corr_matrix.loc['DXY', 'NASDAQ']:.3f}
        - vs S&P500: {corr_matrix.loc['DXY', 'SP500']:.3f}
        """)
    
    with col3:
        st.markdown("#### ⚠️ HY Spread")
        st.info(f"""
        - vs BTC: {corr_matrix.loc['HYSpread', 'BTC']:.3f}
        - vs NASDAQ: {corr_matrix.loc['HYSpread', 'NASDAQ']:.3f}
        - vs S&P500: {corr_matrix.loc['HYSpread', 'SP500']:.3f}
        """)

# ============================================================
# TAB 5: 트레이딩 시그널 (기존 유지)
# ============================================================
with tab5:
    st.header("🎯 현재 트레이딩 시그널")
    st.markdown("**퀀트 3콤보 기반 매매 신호**")
    
    st.markdown("---")
    
    st.subheader("📈 시그널 1: Net Liquidity")
    if netliq_60d > 2:
        st.success(f"""
        ✅ **Net Liquidity 강한 확장** (+{netliq_60d:.2f}%)
        - Fed 유동성 공급 증가
        - 리스크 자산 상승 환경
        - **추천**: BTC/NASDAQ 매수 고려
        """)
    elif netliq_60d < -2:
        st.error(f"""
        ⚠️ **Net Liquidity 강한 축소** ({netliq_60d:.2f}%)
        - Fed 유동성 회수 진행
        - 리스크 자산 하락 압력
        - **추천**: 리스크 자산 매도/경계
        """)
    else:
        st.info(f"""
        ⏸️ **Net Liquidity 중립 구간** ({netliq_60d:+.2f}%)
        - 유동성 변화 미미
        - 다른 요인 주시 필요
        """)
    
    st.markdown("---")
    
    st.subheader("💵 시그널 2: Dollar Index vs Bitcoin")
    if corr_dxy_btc.iloc[-1] < -0.5:
        st.success(f"""
        ✅ **DXY-BTC 강한 역상관** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
        - 달러 약세 = 비트코인 강세
        - **추천**: DXY 하락 시 BTC 매수 기회
        """)
    elif corr_dxy_btc.iloc[-1] > 0:
        st.warning(f"""
        ⚠️ **DXY-BTC 양의 상관** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
        - 비정상적 동행
        - 리스크 회피 모드 가능성
        """)
    else:
        st.info(f"""
        ⏸️ **DXY-BTC 역상관 약화** (상관계수: {corr_dxy_btc.iloc[-1]:.3f})
        - 상관관계 불명확
        - 독립적 움직임
        """)
    
    st.markdown("---")
    
    st.subheader("⚠️ 시그널 3: High Yield Spread")
    if latest['HYSpread'] > 5.0:
        st.error(f"""
        🚨 **HY Spread 위기 임계점 초과** ({latest['HYSpread']:.2f}%)
        - 신용 시장 경색
        - 기업 파산 위험 증가
        - **추천**: 주식 시장 위험! 매도/방어 전략
        """)
    elif latest['HYSpread'] > 4.0:
        st.warning(f"""
        ⚠️ **HY Spread 경계 구간** ({latest['HYSpread']:.2f}%)
        - 신용 위험 상승 중
        - **추천**: 주의 필요, 포지션 축소 고려
        """)
    else:
        st.success(f"""
        ✅ **HY Spread 정상 구간** ({latest['HYSpread']:.2f}%)
        - 신용 시장 안정
        - 주식 시장 건강
        """)
    
    if recent_divergence > 0:
        st.markdown("---")
        st.error(f"""
        🚨 **Divergence 경고**
        - 최근 5일 중 {recent_divergence}일 Divergence 발생
        - S&P 500 상승 + HY Spread 상승
        - 허위 랠리 가능성 (Bear Market Rally)
        - **추천**: 매도 신호, 이익실현 고려
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 종합 신호 점수")
    
    score = 0
    if netliq_60d > 2:
        score += 1
    elif netliq_60d < -2:
        score -= 1
    
    if corr_dxy_btc.iloc[-1] < -0.5:
        score += 1
    elif corr_dxy_btc.iloc[-1] > 0:
        score -= 1
    
    if latest['HYSpread'] < 4.0:
        score += 1
    elif latest['HYSpread'] > 5.0:
        score -= 2
    
    if recent_divergence > 0:
        score -= 1
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if score >= 2:
            st.success(f"""
            ### 🟢 강한 매수 신호
            **점수: +{score}/4**
            - 리스크 온 환경
            - BTC/주식 매수 고려
            """)
        elif score == 1:
            st.info(f"""
            ### 🟡 약한 매수 신호
            **점수: +{score}/4**
            - 중립적 환경
            - 선별적 매수
            """)
        elif score == 0:
            st.warning(f"""
            ### ⚪ 중립 신호
            **점수: {score}/4**
            - 관망 추천
            """)
        elif score == -1:
            st.warning(f"""
            ### 🟡 약한 매도 신호
            **점수: {score}/4**
            - 주의 필요
            - 포지션 축소 고려
            """)
        else:
            st.error(f"""
            ### 🔴 강한 매도 신호
            **점수: {score}/4**
            - 리스크 오프 환경
            - 현금 보유 권장
            """)

# ============================================================
# TAB 6: AI 분석 (기존 유지)
# ============================================================
with tab6:
    st.header("🤖 Gemini AI 분석")
    st.markdown("**Google Gemini 2.0 Flash 기반 시장 분석**")
    
    if not GEMINI_ENABLED:
        st.error("""
        ❌ **Gemini API가 설정되지 않았습니다.**
        
        `.streamlit/secrets.toml` 파일에 다음을 추가하세요:
```toml
        GEMINI_API_KEY = "your-api-key-here"
```
        
        API 키 발급: https://aistudio.google.com/app/apikey
        """)
        st.stop()
    
    st.markdown("---")
    
    # 분석 모드 선택
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "📊 분석 유형 선택",
            ["종합분석", "유동성분석", "달러분석", "신용분석", "트레이딩전략"],
            help="원하는 분석 유형을 선택하세요"
        )
    
    with col2:
        # Deep Dive 모드 토글
        deep_dive_mode = st.toggle(
            "🔬 Deep Dive",
            help="심층 분석 모드: 더 상세하고 깊이 있는 분석을 제공합니다 (응답 시간이 더 걸립니다)"
        )
    
    with col3:
        st.metric("🔋 API 상태", "활성화" if GEMINI_ENABLED else "비활성화")
    
    # Deep Dive 모드 설명
    if deep_dive_mode:
        st.info("""
        🔬 **Deep Dive 모드 활성화**
        
        이 모드에서는 다음과 같은 심층 분석을 제공합니다:
        - 📈 시계열 추세 및 변동성 분석
        - 🎯 다중 시나리오 분석 (Bull/Base/Bear Case)
        - ⚠️ 리스크 매트릭스 및 스트레스 테스트
        - 💰 구체적인 진입/청산 가격 제시
        - 📊 포트폴리오 배분 및 리밸런싱 전략
        - ✅ 실행 가능한 체크리스트
        
        ⏱️ 분석 시간: 약 30-60초 소요
        """)
    
    # 분석 실행 버튼
    button_label = "🚀 Deep Dive 분석 실행" if deep_dive_mode else "🚀 AI 분석 실행"
    
    if st.button(button_label, type="primary", use_container_width=True):
        # 데이터 요약 생성
        data_summary = get_data_summary(df_recent, latest, netliq_60d)
        correlations = get_correlations_summary(corr_matrix)
        signals = get_signals_summary(netliq_60d, latest, corr_dxy_btc.iloc[-1], recent_divergence)
        
        # AI 분석 실행 (모드에 따라 다른 함수 호출)
        if deep_dive_mode:
            analysis_result = analyze_with_gemini_deep_dive(
                analysis_type,
                data_summary,
                correlations,
                signals,
                df_recent,
                latest
            )
            analysis_label = f"Deep Dive {analysis_type}"
        else:
            analysis_result = analyze_with_gemini(
                analysis_type,
                data_summary,
                correlations,
                signals
            )
            analysis_label = analysis_type
        
        # 결과 표시
        st.markdown("---")
        
        # 분석 메타 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 분석 유형", analysis_label)
        with col2:
            st.metric("🔬 모드", "Deep Dive" if deep_dive_mode else "Standard")
        with col3:
            st.metric("⏰ 생성 시각", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown(f"### 📊 {analysis_label} 결과")
        
        # 분석 결과를 박스에 표시
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {"#FF6B35" if deep_dive_mode else "#2E86AB"};'>
            {analysis_result.replace('\n', '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 액션 버튼들
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 다운로드 버튼
            st.download_button(
                "📥 분석 결과 다운로드",
                analysis_result,
                file_name=f"gemini_{analysis_type}_{('deep_dive_' if deep_dive_mode else '')}{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # 저장 버튼
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            
            if st.button("💾 분석 결과 저장", use_container_width=True):
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': analysis_label,
                    'mode': 'Deep Dive' if deep_dive_mode else 'Standard',
                    'result': analysis_result
                })
                st.success("✅ 분석 결과가 저장되었습니다!")
                st.rerun()
        
        with col3:
            # 다른 모드로 재분석
            alt_mode_label = "일반 분석으로" if deep_dive_mode else "Deep Dive로"
            if st.button(f"🔄 {alt_mode_label} 재분석", use_container_width=True):
                st.info(f"💡 토글을 전환하고 다시 분석 버튼을 눌러주세요.")


    # 저장된 분석 히스토리 표시
    if 'analysis_history' in st.session_state and len(st.session_state.analysis_history) > 0:
        st.markdown("---")
        st.markdown("### 📜 분석 히스토리")
        
        # 최근 5개만 표시
        for idx, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
            mode_badge = "🔬 Deep Dive" if item['mode'] == 'Deep Dive' else "📊 Standard"
            with st.expander(f"🕐 {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['type']} ({mode_badge})"):
                st.markdown(item['result'])
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ 히스토리 전체 삭제"):
                st.session_state.analysis_history = []
                st.rerun()
    
    # AI 사용 팁
    st.markdown("---")
    st.markdown("### 💡 AI 분석 활용 가이드")
    
    tab_guide1, tab_guide2, tab_guide3 = st.tabs(["📖 분석 유형별 가이드", "🔬 Deep Dive 가이드", "⚠️ 주의사항"])
    
    with tab_guide1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Standard 분석 (빠른 인사이트)**
            - **종합분석**: 전체 시장 상황 요약 (3-5문장)
            - **유동성분석**: Fed 정책 방향 (2-4문장)
            - **달러분석**: 달러 강도와 영향 (2-4문장)
            - **신용분석**: HY Spread 해석 (2-4문장)
            - **트레이딩전략**: 실행 가능한 전략 (4-6문장)
            
            **추천 상황:**
            - 빠른 시장 체크가 필요할 때
            - 핵심 포인트만 파악하고 싶을 때
            - 매일 아침 브리핑용
            - 5분 이내 빠른 의사결정
            """)
        
        with col2:
            st.markdown("""
            **Deep Dive 분석 (전문적 분석)**
            - **종합분석**: 7가지 관점 심층 분석 (1,500+ 단어)
            - **유동성분석**: Fed 정책 완전 분석 (1,000+ 단어)
            - **달러분석**: 글로벌 매크로 분석 (1,000+ 단어)
            - **신용분석**: 리스크 매트릭스 (1,000+ 단어)
            - **트레이딩전략**: 구체적 진입/청산가 (1,500+ 단어)
            
            **추천 상황:**
            - 중요한 투자 결정 전
            - 주간/월간 전략 수립 시
            - 포트폴리오 리밸런싱 시
            - 심층 리서치가 필요할 때
            """)
    
    with tab_guide2:
        st.markdown("""
        ### 🔬 Deep Dive 분석의 특징
        
        **1. 다중 시나리오 분석**
        - **Bull Case (낙관적)**: 확률 20-30%, 전개 조건, 가격 타겟
        - **Base Case (중립적)**: 확률 40-60%, 가장 가능성 높은 시나리오
        - **Bear Case (비관적)**: 확률 20-30%, 하방 리스크, 방어 전략
        - 각 시나리오별 구체적 대응 액션 플랜
        
        **2. 리스크 매트릭스**
        - **단기 리스크** (1주-1개월): 즉각 대응 필요
        - **중기 리스크** (1-3개월): 모니터링 및 준비
        - **구조적 리스크** (장기): 포트폴리오 구조 조정
        - **Black Swan 이벤트**: 극단적 시나리오 대비
        
        **3. 실행 가능한 트레이딩 전략**
        - **진입가**: 구체적인 가격 레벨 ($XX,XXX)
        - **익절가**: 1차/2차/최종 익절 레벨
        - **손절가**: 명확한 손절 라인
        - **포지션 사이징**: 총 자산 대비 %
        - **리밸런싱 트리거**: 언제 조정할 것인가
        
        **4. 통계 기반 분석**
        - **변동성 분석**: 7일/30일/90일 변동성
        - **상관관계 추세**: 과거 대비 현재 위치
        - **이동평균**: 단기/중기/장기 MA 배열
        - **모멘텀 지표**: 과매수/과매도 판단
        
        **5. 3개월 로드맵**
        - **Week 1-2**: 즉시 실행할 전략
        - **Month 1**: 첫 달 목표와 체크포인트
        - **Month 2-3**: 중기 포지션 조정 계획
        - **주요 이벤트**: FOMC, 경제지표 발표 일정
        
        **6. 일일/주간 체크리스트**
        - 매일 확인할 지표 (Net Liq, DXY, HY Spread)
        - 주간 리뷰 항목 (상관관계, 포트폴리오 성과)
        - 트리거 이벤트 (포지션 변경 조건)
        
        **7. 역사적 패턴 비교**
        - 과거 유사 상황 분석
        - Win Rate / Profit Factor 추정
        - 최대 드로다운 시나리오
        """)
    
    with tab_guide3:
        st.warning("""
        ### ⚠️ 중요한 주의사항
        
        **투자 책임**
        - ❗ AI 분석은 참고용이며, 투자 조언이 아닙니다
        - ❗ 최종 투자 결정은 본인의 책임입니다
        - ❗ 여러 정보원을 종합적으로 검토하세요
        - ❗ 본인의 리스크 허용도를 반드시 고려하세요
        
        **API 사용 제한**
        - 무료 할당량: 일일 1,500 요청, 분당 15 요청
        - Deep Dive 모드: 더 많은 토큰 소비 (Standard의 3-5배)
        - 할당량 초과 시: 24시간 후 재시도 또는 유료 전환
        - 오류 발생 시: 잠시 후 다시 시도
        
        **분석 한계**
        - AI는 과거 데이터 기반으로 학습됨 (2025년 1월까지)
        - 예측 불가능한 이벤트 미고려 (전쟁, 자연재해 등)
        - Black Swan 이벤트 대응 한계
        - 시장은 항상 비이성적일 수 있음
        
        **데이터 시차**
        - FRED 데이터는 1-2일 지연될 수 있음
        - 실시간 급변 상황 반영 어려움
        - 최신 뉴스와 교차 검증 필수
        - 주말/공휴일 데이터 업데이트 없음
        
        **AI의 한계**
        - 확률적 추론이므로 100% 정확도 보장 안 됨
        - 동일한 입력에도 다른 결과 가능
        - 맥락 이해 한계 있음
        - 창의적 해석은 제한적
        """)
    
    # 대화형 챗봇 섹션
    st.markdown("---")
    st.markdown("### 💬 AI와 대화하기")
    st.caption("궁금한 점을 자유롭게 물어보세요. AI가 현재 시장 데이터를 바탕으로 답변합니다.")
    
    # 세션 상태 초기화
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # 예시 질문 버튼
    st.markdown("**💡 예시 질문:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🤔 지금 비트코인 사도 될까요?", use_container_width=True):
            example_prompt = "현재 시장 상황에서 비트코인을 매수해도 괜찮을까요? 리스크는 무엇인가요?"
            st.session_state.example_prompt = example_prompt
    
    with example_col2:
        if st.button("📊 포트폴리오 비중 추천", use_container_width=True):
            example_prompt = "현재 상황에서 BTC, 주식, 현금 비중을 어떻게 가져가면 좋을까요?"
            st.session_state.example_prompt = example_prompt
    
    with example_col3:
        if st.button("⚠️ 현재 가장 큰 리스크는?", use_container_width=True):
            example_prompt = "지금 시장에서 가장 주의해야 할 리스크 요인은 무엇인가요?"
            st.session_state.example_prompt = example_prompt
    
    # 대화 초기화 버튼
    if len(st.session_state.chat_messages) > 0:
        if st.button("🔄 대화 초기화", type="secondary"):
            st.session_state.chat_messages = []
            if 'example_prompt' in st.session_state:
                del st.session_state.example_prompt
            st.rerun()
    
    # 대화 히스토리 표시
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 (예시 질문이 있으면 자동 입력)
    default_prompt = st.session_state.get('example_prompt', '')
    if default_prompt and 'example_prompt' in st.session_state:
        del st.session_state.example_prompt
    
    if prompt := st.chat_input("질문을 입력하세요...", key="chat_input"):
        # 사용자 메시지 추가
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 컨텍스트 정보 추가
        context = f"""
당신은 20년 경력의 거시경제 및 퀀트 투자 전문가입니다.
현재 사용자와 대화 중이며, 아래 최신 시장 데이터를 기반으로 답변해주세요.

## 현재 시장 상황 (최신 데이터)
{get_data_summary(df_recent, latest, netliq_60d)}

## 주요 상관관계
{get_correlations_summary(corr_matrix)}

## 현재 시그널 상태
{get_signals_summary(netliq_60d, latest, corr_dxy_btc.iloc[-1], recent_divergence)}

## 사용자 질문
{prompt}

## 답변 가이드라인
1. 친절하고 전문적으로 답변하세요
2. 위 데이터를 적극 활용하여 구체적으로 답변하세요
3. 필요시 숫자와 %를 명시하세요
4. 3-5문장으로 간결하게 답변하세요
5. 투자 조언이 아닌 참고 정보임을 명시하세요
6. 확실하지 않은 것은 솔직히 인정하세요
"""
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("💭 생각 중..."):
                try:
                    response = gemini_model.generate_content(context)
                    answer = response.text
                    st.markdown(answer)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    error_msg = f"❌ 오류가 발생했습니다: {str(e)}\n\n💡 무료 할당량을 초과했거나 일시적인 문제일 수 있습니다. 잠시 후 다시 시도해주세요."
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # 자동으로 예시 프롬프트 처리
    elif default_prompt:
        # 사용자 메시지 추가
        st.session_state.chat_messages.append({
            "role": "user",
            "content": default_prompt
        })
        st.rerun()
    
    # 대화 통계
    if len(st.session_state.chat_messages) > 0:
        st.markdown("---")
        total_messages = len(st.session_state.chat_messages)
        user_messages = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("💬 총 메시지", total_messages)
        with stat_col2:
            st.metric("👤 사용자 질문", user_messages)
        with stat_col3:
            st.metric("🤖 AI 답변", total_messages - user_messages)
