import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock

# 오늘 날짜 가져오기
today = datetime.now()
formatted_today = today.strftime('%Y%m%d')

# 1년 전 날짜 계산
one_year_ago = today - timedelta(days=365)
formatted_one_year_ago = one_year_ago.strftime('%Y%m%d')

# df에 ohlcv 저장
df = stock.get_market_ohlcv_by_date(formatted_one_year_ago, formatted_today, "005930")

# 결과 값을 계산하여 새로운 데이터프레임 생성
result_df = df['등락률'].apply(lambda x: 1 if x >= 0 else 0)

# 일자별 Div/BPS/PER/EPS 조회
mjm = stock.get_market_fundamental(formatted_one_year_ago, formatted_today, "005930")

# 최종 데이터프레임 생성
final_df = pd.DataFrame({
    'result': result_df,
    '거래량': df['거래량'],
    'PER': mjm['PER'],
    'DIV': mjm['DIV']
})

final_df.index = final_df.index.strftime('%Y%m%d')

# 365일 동안 반복하여 순매수 데이터 가져오기
for i in range(365):
    # 날짜 계산
    date_for_data = today - timedelta(days=i)
    date_for_data_str = date_for_data.strftime('%Y%m%d')
    date_for_data1 = date_for_data - timedelta(days=1)
    date_for_data1_str = date_for_data1.strftime('%Y%m%d')

    try:
        # 데이터 가져오기
        gattcha = stock.get_market_net_purchases_of_equities(date_for_data1_str, date_for_data_str, "KOSPI", "개인")

        # gattcha가 비어 있는지 확인하고 비어 있으면 무시
        if gattcha.empty:
            continue  # 데이터가 없으면 다음 반복으로 넘어감

        # gattcha 구조 확인
        print(f"{date_for_data_str}에 대한 gattcha 데이터:")

        # '티커' 열이 있을 경우 필터링
        if '티커' in gattcha.columns:
            gattcha['티커'] = gattcha['티커'].astype(str)  # 티커를 문자열로 변환
            gattcha_filtered = gattcha[gattcha['티커'] == '005930']  # 티커를 문자열로 사용
        else:
            # 인덱스를 문자열로 변환하여 필터링
            gattcha_filtered = gattcha[gattcha.index.astype(str) == '005930']

        # 날짜 열 추가 및 인덱스로 설정
        gattcha_filtered['날짜'] = date_for_data_str  # 날짜 열 추가
        gattcha_filtered.set_index('날짜', inplace=True)  # 날짜를 인덱스로 설정

        # 기존 데이터에 추가된 gattcha_filtered 병합
        final_df = final_df.combine_first(gattcha_filtered)

    except Exception as e:
        print(f"데이터 가져오기 오류: {e}")
