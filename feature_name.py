target = 'purchased_flag' #label

numerical_feature = [
    'purchases', #行为数据
    'open_purchases_daily_30', #统计类
    'open_purchases_daily_pct_30', #统计类
    'open_purchases_daily_180', #统计类
    'open_purchases_daily_pct_180', #统计类
    'avg_time_between_open_purchases', #统计类
    'avg_open_purchase_volume', #统计类
    'days_since_last_open_purchase_60',  #统计类
    'open_purchase_dts_60', #统计类
    'open_purchase_weeks_60', #统计类
    'days_until_expected_open_purchase', #统计类
    'open_purchases_30', #统计类
    'open_purchases_90', #统计类
    'view_vins_1', #? 统计类
    'total_views_1', #? 统计类
    'view_dates_1', #? 统计类
    'view_vins_3', #? 统计类
    'total_views_3',#? 统计类
    'view_dates_3', #? 统计类
    'view_vins_7', #? 统计类
    'total_views_7', #? 统计类
    'view_dates_7', #? 统计类
    'bid_vins_1', #? 统计类
    'total_bids_1', #? 统计类
    'bid_dates_1', #? 统计类
    'bid_vins_3',#? 统计类
    'total_bids_3', #? 统计类
    'bid_dates_3', #? 统计类
    'bid_vins_7', #? 统计类
    'total_bids_7',#? 统计类
    'bid_dates_7', #? 统计类
    'lot_size',  #context
    'avg_lot_size_90', #统计类
    'relative_lot_size', #relative_lot_size
    'cars_moved_off_retail_7', #? 统计类
    'cars_moved_off_retail_30',#? 统计类
    'cars_moved_off_retail_90', #? 统计类
]

categorical_feature = [
    'dow', #上下文 时间
    'purchase_date', #上下文 时间
    'buyer_cluster', #标签类 
    'cluster_G', #标签类 
    'cluster_NA', #标签类 
    'cluster_NG',#标签类 
    'cluster_O', #标签类 
    'cluster_P+', #标签类 
    'cluster_P-', #标签类 
    'cluster_SC', #标签类 
    'cluster_Super',#标签类 
    'buyer_id_masked' # 用户id
]
