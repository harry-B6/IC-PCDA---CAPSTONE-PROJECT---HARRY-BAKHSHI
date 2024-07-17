import statistics
import pickle
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
#subfunction to fill new df:
def panel_to_ts_janatahack(i):
    #^https://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    #^https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    #Both accessed 15/06/2024
    file = '/Users/harrybakhshi/Desktop/Python_notes/data/PCDA_capstone/janatahack_demand_forecasting_store_product_list.pik'
    # with open(file, 'wb') as f:
    #     pickle.dump(store_product_list, f) #write df to .pik file on disk
    with open(file, 'rb') as f:
         store_product_list = pickle.load(f) #load pickle file 'file' into variable
    file = '/Users/harrybakhshi/Desktop/Python_notes/data/PCDA_capstone/janatahack_demand_forecasting_data__ts_df.pik'
    # with open(file, 'wb') as f:
    #     pickle.dump(_ts_df, f) #write df to .pik file on disk
    with open(file, 'rb') as f:
         _ts_df = pickle.load(f) #load pickle file 'file' into variable
    ts_df = _ts_df
    file = '/Users/harrybakhshi/Desktop/Python_notes/data/PCDA_capstone/janatahack_demand_forecasting_data_df.pik'
    # with open(file, 'wb') as f:
    #     pickle.dump(df, f) #write df to .pik file on disk
    with open(file, 'rb') as f:
         df = pickle.load(f) #load pickle file 'file' into variable    
    col_list = ['total_price', 'base_price', 'units_sold', 'is_featured_sku', 'is_display_sku']
    col_list_2 = ['total_price', 'base_price', 'units_sold', 'percent_featured', 'percent_display']
    store = store_product_list[i][0:4]
    product = store_product_list[i][5:11]
    #^https://www.freecodecamp.org/news/how-to-substring-a-string-in-python/
    #Accessed 15/06/2024
    # for timestamp in range(1):
    for timestamp in range(len(ts_df)):
        for col in range(len(col_list)):
            #Create lists for averaging - see lower code
            sales_price_list_per_ts_per_store_per_product = []
            base_price_list_per_ts_per_store_per_product = []
            units_sold_list_per_ts_per_store_per_product = []
            is_featured_list_per_ts_per_store_per_product = []
            is_display_list_per_ts_per_store_per_product = []
            # for row in df.index:
            #     if df['timestamp'][row] == ts_df['timestamp'][timestamp]:
            for row in df.index:
                if df['timestamp'][row] == ts_df['timestamp'][timestamp]:
                    row1 = row
                    break
            for row in range(len(df[row:])):
                if df['timestamp'][row] != ts_df['timestamp'][timestamp]:
                    row2 = row
                    break
            for row in range(len(df[row1:row2])):
                #leave space
                if df['store_id'][row] == int(store):
                    if df['sku_id'][row] == int(product):
                        #Append to lists for averaging - see lower code
                        if col_list[col] == 'total_price':
                            sales_price_list_per_ts_per_store_per_product.append(df[col_list[col]][row])
                        if col_list[col] == 'base_price':
                            base_price_list_per_ts_per_store_per_product.append(df[col_list[col]][row])
                        if col_list[col] == 'units_sold':
                            units_sold_list_per_ts_per_store_per_product.append(df[col_list[col]][row])
                        if col_list[col] == 'is_featured_sku':
                            is_featured_list_per_ts_per_store_per_product.append(df[col_list[col]][row])
                        if col_list[col] == 'is_display_sku':
                            is_display_list_per_ts_per_store_per_product.append(df[col_list[col]][row])
            #In case sales of the same product in different stores at the timestamp take the average for the product at that timestamp:
            if col_list[col] == 'total_price':
                if not sales_price_list_per_ts_per_store_per_product:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = 0
                else:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = statistics.median(sales_price_list_per_ts_per_store_per_product)  
            if col_list[col] == 'base_price':
                if not base_price_list_per_ts_per_store_per_product:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = 0
                else:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = statistics.median(base_price_list_per_ts_per_store_per_product)  
            if col_list[col] == 'units_sold':
                if not units_sold_list_per_ts_per_store_per_product:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = 0
                else:
                    ts_df[(store + '_' + product + ('median_' + col_list[col]))][timestamp] = statistics.median(units_sold_list_per_ts_per_store_per_product)  
            
            #can't use mode or median as instances of is_featured_sku or is_display_sku can be too 
            #infrequent to detect the existing featuring or displaying in this way in binning multiple retailings of an individual 
            #product in all
            #the different stores per timestamp - use mean; as is_featured count / all product observations per timestamp,
            #is_display count / all product observations per timestamp
            #Measure of how much featuring a product/displaying it was emphasised at that timestamp

            #Use this rather than sum because describes no of values in bin as well as no featured, whereas sum does not; 
            #as mean ok because keep outliers in potentially skewed bin input value distributions (which as discrete 
            #distributions can of course be skewed and have a mean) by Grubbs 1969 definition of an outlier (1), 
            #because after considering the cause of the outliers and obviously eliminating errors, if trying to model a 
            #population that includes these outliers, it is important to include them, as they are part of the population (2)
            #(1) - Module 12, IC PCDA: ‘Required discussion 12.3: Anomaly detection resources’ - Harry Bakhshi. Accessed 
            #18/05/2024.
            #(2) - Module 12, IC PCDA: 'Mini-lesson 12.2: Consequences of removing outliers'. Accessed 09/06/2024
            
            if col_list_2[col] == 'percent_featured':
                if not is_featured_list_per_ts_per_store_per_product:
                    ts_df[(store + '_' + product + ('percent_featured'))][timestamp] = 0
                else:
                    ts_df[(store + '_' + product + ('percent_featured'))][timestamp] = sum(is_featured_list_per_ts_per_store_per_product) / len(is_featured_list_per_ts_per_store_per_product)  
            if col_list_2[col] == 'percent_display':
                if not is_featured_list_per_ts_per_store_per_product:
                    ts_df[(store + '_' + product + ('percent_display'))][timestamp] = 0
                else:
                    ts_df[(store + '_' + product + ('percent_display'))][timestamp] = sum(is_display_list_per_ts_per_store_per_product) / len(is_display_list_per_ts_per_store_per_product)
    return ts_df[store_product_list[i]]