import zipfile
import warnings
import pandas as pd
from py2neo import Graph
from datetime import datetime
import pyodbc
import pandas as pd
import spacy
from concurrent.futures import ThreadPoolExecutor
import time
from py2neo import Graph
import pandas as pd
import networkx as nx
# Suppress all warnings
warnings.filterwarnings('ignore')
from py2neo.errors import ServiceUnavailable
def check_connection(uri, username, password):
    try:
        graph = Graph(uri, auth=(username, password))
        # Run a simple query to check the connection
        graph.run("RETURN 1")
        print("Connection successful")
        return graph
    except ServiceUnavailable:
        print("Connection failed")
        return None

uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "123456789"
from bench_mark_help import check_record_exists_query_neo4j,check_record_exists_query_neo4j_2,check_record_exists_query_neo4j_2_1,check_record_exists_query_neo4j_3_1,check_record_exists_query_neo4j_3_2,check_record_exists_query_neo4j_4 
# x=['|43249|',	'|32140|',	'|50256|',	'|43569|',	'|50563|',	'|37787|',	'|30515|',	'|53836|',	'|51820|',	'|49939|',	'|37546|',	'|30771|',	'|31490|',	'|49759|',	'|43389|',	'|52040|',	'|39840|',	'|26421|',	'|49080|',	'|34650|',	'|41582|',	'|28113|',	'|46760|',	'|50559|',	'|31364|',	'|40961|',	'|45671|',	'|45335|',	'|49611|',	'|46225|',	'|43418|',	'|51377|',	'|27943|',	'|45475|',	'|54689|',	'|50788|',	'|25486|',	'|45812|',	'|36878|',	'|49268|',	'|27560|',	'|44038|',	'|36206|',	'|44005|',	'|30334|',	'|41901|',	'|46960|',	'|50193|',	'|52724|',	'|40960|',	'|46798|',	'|51995|',	'|29305|',	'|51893|',	'|47347|',	'|31098|',	'|29298|',	'|25614|',	'|41787|',	'|51613|',	'|39530|',	'|47802|',	'|31186|',	'|36256|',	'|30380|',	'|51173|',	'|45570|',	'|33515|',	'|33047|',	'|53019|',	'|26983|',	'|49741|',	'|27189|',	'|28691|',	'|50994|',	'|47910|',	'|49566|',	'|51793|',	'|51639|',	'|43391|',	'|30894|',	'|27243|',	'|50714|',	'|31486|',	'|46584|',	'|47013|',	'|37514|',	'|52780|',	'|37806|',	'|46304|',	'|54560|',	'|51958|',	'|39217|',	'|49135|',	'|46824|',	'|47581|',	'|28927|',	'|39611|',	'|39111|',	'|33906|',	'|36001|',	'|41006|',	'|48606|',	'|49668|',	'|47415|',	'|33532|',	'|50862|',	'|47922|',	'|30188|',	'|38388|',	'|27381|',	'|40554|',	'|47018|',	'|47426|',	'|51606|',	'|54480|',	'|54378|',	'|26416|',	'|49187|',	'|50014|',	'|48277|',	'|29526|',	'|48656|',	'|46030|',	'|46283|',	'|50426|',	'|45308|',	'|46721|',	'|45633|',	'|24625|',	'|52156|',	'|46775|',	'|53868|',	'|50554|',	'|52105|',	'|51300|',	'|38254|',	'|50478|',	'|38256|',	'|51332|',	'|45453|',	'|33591|',	'|38282|',	'|50499|',	'|44626|',	'|49942|',	'|47275|',	'|47477|',	'|38785|',	'|54115|',	'|45389|',	'|21478|',	'|46139|',	'|38767|',	'|44999|',	'|36085|',	'|41556|',	'|29564|',	'|41417|',	'|48019|',	'|47501|',	'|42050|',	'|43645|',	'|46467|',	'|40987|',	'|44788|',	'|32890|',	'|37141|',	'|33713|',	'|47779|',	'|48776|',	'|49987|',	'|30412|',	'|51101|',	'|25821|',	'|26803|',	'|47936|',	'|47879|',	'|49561|',	'|30330|',	'|53069|',	'|51070|',	'|45770|',	'|47966|',	'|37350|',	'|47350|',	'|49455|',	'|49501|',	'|36947|',	'|30556|',	'|37243|',	'|30228|',	'|45461|',	'|25426|',	'|54527|',	'|37700|',	'|52338|',	'|43690|']
# output_file_path = r'D:\2020_test\benchmark.txt'
# # Path to the ZIP file
# zip_file_path = r'D:\2020.zip'

# header="Date|Time|Reg No|Terminal Id|System|Program|User Group|Trans Type|Object Type|Record Key|Record Value\n"

# encoding = 'ISO-8859-1'
# import re
# search_pattern = re.compile('|'.join(map(re.escape, x)))
# output_encoding = 'utf-8'

# import time
# start_time = time.time()
# print(start_time)

# # with zipfile.ZipFile(zip_file_path, 'r') as z, open(output_file_path, 'w', buffering=1, encoding=output_encoding, errors='ignore') as output_file:
# #     output_file.write(header)
# #     for file_name in z.namelist():
# #         if file_name.endswith('.txt'):
# #             print(f"Processing {file_name}...")
# #             with z.open(file_name) as f:
# #                 for line in f:
# #                     decoded_line = line.decode(encoding, errors='ignore')
# #                     if search_pattern.search(decoded_line):
# #                         output_file.write(decoded_line.strip() + '\n')

# #I will commnet thsi part as I am using data form actual cops which are txt that chnage to csv start here
# # from datetime import datetime
# # def transform_date_format(date_str):
# #     # Determining if the date is valid based on its length
# #         if len(date_str) == 7:  # Day is two digits
# #             day = date_str[:1]
# #             month = date_str[1:3]
# #             year = date_str[3:]
# #         else:  
# #             day = date_str[:2]
# #             month = date_str[2:4]
# #             year = date_str[4:]
# #         return datetime.strptime(year+'-'+month+'-'+day,'%Y-%m-%d').date()

# # # Apply the function, handling potential errors
# # def apply_with_error_handling(df, func):
# #   def wrapper(row):
# #     try:
# #       return func(row)
# #     except Exception as e:
# #       # Handle the exception (e.g., log the error, return a default value)
# #       print(f"Error processing row {row}: {e}")
# #       return None  # Or any other appropriate value for skipped rows
# #   return df['Date'].apply(wrapper)


# # def try_parse_datetime(row):
# #   """Attempts to parse the time string and returns a datetime object.

# #   Args:
# #       row: A pandas Series containing 'Date' and 'Time' columns.

# #   Returns:
# #       A datetime object if parsing is successful, None otherwise.
# #   """
# #   try:
# #     # Use a more flexible format string to handle potential variations
# #     date_time = datetime.combine(row['Date'], datetime.strptime(row['Time'], "%H:%M:%S").time())
# #     return date_time
# #   except ValueError:
# #     # Handle the parsing error (e.g., log the error, return None)
# #     print(f"Error parsing datetime for row {row.name}: {row['Time']}")
# #     return None


# # #very simialr to next code but this is just to pre process one single officer log file
# # import zipfile
# # import pandas as pd
# # import io
# # import os
# # import csv

# # # Directory containing the .txt files
# # directory = r'D:\2020_test'

# # # Iterate over each file in the directory
# # output_csv_path = 'Program_Type_All.csv'
# # with open(output_csv_path, 'w', newline='') as outfile:
# #     csv_writer = csv.writer(outfile)
# #     # Write the header row
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.txt'):
# #             file_path = os.path.join(directory, filename)
# #             # Try to open the file with different encodings
# #             df = pd.read_csv(file_path, sep='|',on_bad_lines='skip',engine="python",encoding='ISO-8859-1')
# #             df = df.astype(str)
# #             #df[0] = df[0].apply(str)
# #             df = df[(df.iloc[:, 0].astype(str).str.len() == 7) | (df.iloc[:, 0].astype(str).str.len() == 8)]
# #             df = df.dropna(subset=[df.columns[5]])#for rows where in txt file they are move to next line without any addional information
# #             #df['Date'] = df['Date'].apply(transform_date_format)
# #             df['Date'] = apply_with_error_handling(df.copy(), transform_date_format)
# #             from datetime import datetime
# #             #df['DateTime'] = df.apply(lambda row: datetime.combine(row['Date'], datetime.strptime(row['Time'], "%H:%M:%S").time()), axis=1)
# #             df['DateTime'] = df.apply(try_parse_datetime, axis=1)
# #             strings_to_remove = ["UNKNOWN", "UNKNOW", "UNKOWN", "UNK", "UNKNWN"]
# #             df = df[~df['Record Key'].isin(strings_to_remove)]
# #             file_exists = os.path.isfile(output_csv_path)
# #             df.to_csv(output_csv_path, mode='a', sep=',', index=False)


# #I will commnet thsi part as I am using data form actual cops which are txt that chnage to csv end here

#this is necessry to chnage teh format of columns


event_cni_relationships_INV_PARTY=pd.read_csv(r'D:\Python\invparty\event_cni_relationships_INV_PARTY.csv')
off_event_relationships_ADMIN_LOG=pd.read_csv(r'D:\Python\adminlog\neo4j_relationships_ADMIN_LOG.csv')
off_event_relationships_INV_PARTY=pd.read_csv(r'D:\Python\invparty\off_event_relationships_INV_PARTY.csv')
off_event_relationships_CHARGE=pd.read_csv(r'D:\Python\charge\off_event_relationships_CHARGE.csv')
event_cni_relationships_CHARGE=pd.read_csv(r'D:\Python\charge\event_cni_relationships_CHARGE.csv')
off_cni_relationships_CHARGE=pd.read_csv(r'D:\Python\charge\off_cni_relationships_CHARGE.csv')
event_cni_relationships_PARTY_LINK=pd.read_csv(r'D:\Python\cop_party_link\event_cni_relationships_PARTY_LINK.csv')
off_event_relationships_PARTY_LINK=pd.read_csv(r'D:\Python\cop_party_link\off_event_relationships_PARTY_LINK.csv')
off_event_relationships_EVENT=pd.read_csv(r'D:\Python\event\officer_event_EVENT.csv')
off_cni_relationships_ACTION=pd.read_csv(r'D:\Python\copaction\officer_cni__ACTION.csv')
off_event_relationships_ACTION=pd.read_csv(r'D:\Python\copaction\officer_event__ACTION.csv')
off_cni_relationships_COP_PARTY=pd.read_csv(r'D:\Python\copparty\officer_cni_COP_PARTY.csv')
off_event_relationships_NARRATIVE=pd.read_csv(r'D:\Python\narrative\officer_event_NARRATIVE.csv')
event_cni_relationships_maingraph=pd.read_csv(r'D:\Python\\maingraph\event_CENTRAL_NAMES_REF_NUM_MAINGRAPH.csv')
event_event_location_relationships_maingraph=pd.read_csv(r'D:\Python\\maingraph\event_event_location_MAINGRAPH.csv')
cni_party_location_relationships_maingraph=pd.read_csv(r'D:\Python\\maingraph\CENTRAL_NAMES_REF_NUM_party_location_MAINGRAPH.csv')
evenet_vehicle_relationships_COP_maingraph=pd.read_csv(r'D:\Python\\maingraph\event_VEH_REF_NUM_MAINGRAPH.csv')
veh_num_plate_relationships_maingraph=pd.read_csv(r'D:\Python\\maingraph\VEH_REF_NUM_PLATE_NUM_MAINGRAPH.csv')


def compare_and_assign_group(df, column_name):
    """
    Compares two consecutive rows in a specified column and assigns a group based on similarity.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compare.

    Returns:
        pd.Series: A new Series containing the group assignments.
    """

    groups = [1]  # Start with group 1 for the first row
    prev_set = df.iloc[0][column_name].split(",")  # Initialize a variable to store the previous set
    

    for i in range(1, len(df)):
        # Use the previous set if available, else create a new set from the current row
        current_set = set(int(num) for num in df.iloc[i][column_name].split(","))

        # Check for any elements in common with the previous set
        if bool(current_set.intersection(prev_set)):
            groups.append(groups[-1])  # Keep the same group as the previous row
        else:
            groups.append(groups[-1] + 1)  # Increment group counter and assign new group

        prev_set = current_set  # Update the previous set for the next iteration

    return pd.Series(groups, index=df.index, name='group')

import gc

def cluster_creation(df):
    graph = Graph("bolt://127.0.0.1:7687",name="maingraph", auth=("neo4j", "123456789"))
    dff=df[df['Object Type'].isin(['PARTY ENQUIRY','LEGAL PROCESS ENQUIRY','CRIMINAL RECORD ENQUIRY','CRIMINAL HISTORY REPORT'])]
    node_ids = dff['Record Key'].tolist()
    node_ids=list(set(node_ids))
    # Execute the Cypher query
    query = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE x.cniId in {node_ids}
    RETURN apoc.map.removeKeys(properties(n), ['cniId']) AS n_properties, x.cniId
    """
    result_party = graph.run(query).data()
    result_party = [(list(d['n_properties'].values())[0], d['x.cniId']) for d in result_party]

    graph = Graph("bolt://127.0.0.1:7687",name="maingraph", auth=("neo4j", "123456789"))
    dff=df[df['Object Type']=='EVENT ENQUIRY']
    node_ids = dff['Record Key'].tolist()

    # Execute the Cypher query
    query = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.eventId in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.eventId']) AS x_properties, n.eventId

    """
    result_event = graph.run(query).data()
    result_event = [(list(d['x_properties'].values())[0], d['n.eventId']) for d in result_event]


    dff=df[df['Object Type']=='LOCATION ENQUIRY']
    node_ids = dff['Record Key'].tolist()

    # Execute the Cypher query
    query1 = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.party_location in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.party_location']) AS x_properties, n.party_location
    """
    query2 = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.event_location in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.event_location']) AS x_properties, n.event_location
    """
    result_location = graph.run(query1).data()
    result_location = [(list(d['x_properties'].values())[0], d['n.party_location']) for d in result_location]
    result_location2 = graph.run(query2).data()
    result_location.extend([(list(d['x_properties'].values())[0], d['n.event_location']) for d in result_location2])

  
    dff=df[df['Object Type']=='VEHICLE ENQUIRY']
    node_ids = dff['Record Key'].tolist()

    def process_value(value):
        # If the value contains a space, return everything after the space
        if ' ' in value:
            return value.split(' ', 1)[1]  # Split at the first space and return everything after it
        else:
            return value  # Return the value unchanged

    node_ids = [process_value(value) for value in node_ids]


    # Execute the Cypher query
    query1 = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.VEH_REF_NUM in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.VEH_REF_NUM']) AS x_properties, n.VEH_REF_NUM
    """
    query2 = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.PLATE_NUM in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.PLATE_NUM']) AS x_properties, n.PLATE_NUM
    """
    result_vehicle = graph.run(query1).data()
    result_vehicle = [(list(d['x_properties'].values())[0], d['n.VEH_REF_NUM']) for d in result_vehicle]
    result_vehicle2 = graph.run(query2).data()
    result_vehicle.extend([(list(d['x_properties'].values())[0], d['n.PLATE_NUM']) for d in result_vehicle2])

    dff=df[df['Object Type']=='INFORMATION REPORT ENQUIRY']
    node_ids = dff['Record Key'].tolist()

    # Execute the Cypher query
    query = f"""
    MATCH (n)-[r:RELATED]-(x)
    WHERE n.eventId in {node_ids}
    RETURN apoc.map.removeKeys(properties(x), ['n.eventId']) AS x_properties, n.eventId

    """
    result_information_enqury = graph.run(query).data()
    result_information_enqury = [(list(d['x_properties'].values())[0], d['n.eventId']) for d in result_information_enqury]


    filtered_dict=result_event
    filtered_dict.extend(result_location)
    filtered_dict.extend(result_party)
    filtered_dict.extend(result_vehicle)
    filtered_dict.extend(result_information_enqury)

    node_ids = list(set(df['Record Key'].tolist()))
    node_ids = [item for item in node_ids if item != '0']

    Gx = nx.Graph()
    Gx.add_edges_from(filtered_dict)

    Dict_netwrok={}
    cluster=1
    for z in node_ids:
        try:
            shortest_paths = nx.single_source_shortest_path_length(Gx, source=z, cutoff=2)
            for x in [node for node, distance in shortest_paths.items()]:
                if x in Dict_netwrok:
                    Dict_netwrok[x].append(cluster)
                else:
                    Dict_netwrok[x]=[cluster]
            cluster=cluster+1
        except:
            pass


    def merge_values(lst):
        try:
            return ', '.join(map(str, lst))
        except:
            return ''

    import csv
    Dict_netwrok={key:Dict_netwrok.get(key, None) for key in node_ids}
    print('almost finsh with group generation')
    with open('multi_cluster_per_node.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in Dict_netwrok.items():
            try:
                writer.writerow([key, merge_values(value)])
            except:
                pass

    import pandas as pd

    df2=pd.read_csv('multi_cluster_per_node.csv',names=['Record Key', 'dict_value'],header=None,encoding='ISO-8859-1')
    merged_df = df.reset_index().merge(df2, on='Record Key', how='left').set_index('index')
    print('merge done')


    import gc
    del df
    # Force garbage collection
    gc.collect()
    merged_df.dropna(inplace=True,subset = ['dict_value'])
    merged_df=merged_df[merged_df['Trans Type']!='Z']
    merged_df.reset_index(drop=True, inplace=True)
    merged_df['group']=0
    merged_df['group'] = compare_and_assign_group(merged_df, 'dict_value')

    #################################
    #################################This line will filter all the rego for admin peopel and I have to comment this , in he fture I might activate it again
    #################################I am not planning to use this for admin people so I always filter very admin in my code
    merged_df=merged_df[merged_df['Reg No'].astype(str).str.len()<=6]



    counts = merged_df.groupby('group')['Record Key'].nunique()

    # Filter out groups where the count is equal to 1
    groups_to_remove = counts[counts == 1].index

    # Remove rows corresponding to those groups from the original DataFrame
    merged_df_filtered_df  = merged_df[~merged_df['group'].isin(groups_to_remove)].copy()
    merged_df_filtered_df['group'] = compare_and_assign_group(merged_df_filtered_df, 'dict_value')
    df=merged_df.copy()
    #if I am using neo4j the chnage of reg no to str does make sense, but now that I am moing from neo4j this stage will mess all my future calcuation so I comment this one for now
        #**********************************************************
    #**********************************************************
    #**********************************************************
    #**********************************************************
    #**********************************************************
    #**********************************************************
    #df['Reg No']=df['Reg No'].astype(str)
        #**********************************************************
    #**********************************************************
    #**********************************************************
        #**********************************************************
    #**********************************************************
    #**********************************************************

    #df['Date']= pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    del graph
    # Trigger garbage collection
    gc.collect()

    return df

def cluster_creation_no_neo(df):
    dff=df[df['Object Type'].isin(['PARTY ENQUIRY','LEGAL PROCESS ENQUIRY','CRIMINAL RECORD ENQUIRY','CRIMINAL HISTORY REPORT'])]
    node_ids = dff['Record Key'].tolist()
    node_ids=list(set(node_ids))
    # Execute the Cypher query
    df1=event_cni_relationships_maingraph[event_cni_relationships_maingraph[':END_ID(CNI)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_cni_relationships_maingraph.columns[0]: 'source', event_cni_relationships_maingraph.columns[1]: 'target'})
    df2=cni_party_location_relationships_maingraph[cni_party_location_relationships_maingraph[':START_ID(CNI)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={cni_party_location_relationships_maingraph.columns[0]: 'source', cni_party_location_relationships_maingraph.columns[1]: 'target'})
    result_party = pd.concat([df1, df2], axis=0)
    

    dff=df[df['Object Type']=='EVENT ENQUIRY']
    node_ids = dff['Record Key'].tolist()
    df1=event_cni_relationships_maingraph[event_cni_relationships_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_cni_relationships_maingraph.columns[0]: 'source', event_cni_relationships_maingraph.columns[1]: 'target'})
    df2=event_event_location_relationships_maingraph[event_event_location_relationships_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_event_location_relationships_maingraph.columns[0]: 'source', event_event_location_relationships_maingraph.columns[1]: 'target'})
    df3=evenet_vehicle_relationships_COP_maingraph[evenet_vehicle_relationships_COP_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={evenet_vehicle_relationships_COP_maingraph.columns[0]: 'source', evenet_vehicle_relationships_COP_maingraph.columns[1]: 'target'})
    result_event=pd.concat([df1, df2,df3], axis=0)
    

    dff=df[df['Object Type']=='LOCATION ENQUIRY']
    node_ids = dff['Record Key'].tolist()
    result_location = event_event_location_relationships_maingraph[event_event_location_relationships_maingraph[':END_ID(event_location)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_event_location_relationships_maingraph.columns[0]: 'source', event_event_location_relationships_maingraph.columns[1]: 'target'})
    result_location2 = cni_party_location_relationships_maingraph[cni_party_location_relationships_maingraph[':END_ID(party_location)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={cni_party_location_relationships_maingraph.columns[0]: 'source', cni_party_location_relationships_maingraph.columns[1]: 'target'})
    result_location=pd.concat([result_location, result_location2], axis=0)

  

    dff=df[df['Object Type']=='VEHICLE ENQUIRY']
    node_ids = dff['Record Key'].tolist()
    def process_value(value):
        # If the value contains a space, return everything after the space
        if ' ' in value:
            return value.split(' ', 1)[1]  # Split at the first space and return everything after it
        else:
            return value  # Return the value unchanged

    node_ids = [process_value(value) for value in node_ids]
    result_vehicle = evenet_vehicle_relationships_COP_maingraph[evenet_vehicle_relationships_COP_maingraph[':END_ID(VEH_REF_NUM)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={evenet_vehicle_relationships_COP_maingraph.columns[0]: 'source', evenet_vehicle_relationships_COP_maingraph.columns[1]: 'target'})
    result_vehicle2 = veh_num_plate_relationships_maingraph[veh_num_plate_relationships_maingraph[':END_ID(PLATE_NUM)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={veh_num_plate_relationships_maingraph.columns[0]: 'source', veh_num_plate_relationships_maingraph.columns[1]: 'target'})
    result_vehicle=pd.concat([result_vehicle, result_vehicle2], axis=0)


    dff=df[df['Object Type']=='INFORMATION REPORT ENQUIRY']
    node_ids = dff['Record Key'].tolist()

    df1=event_cni_relationships_maingraph[event_cni_relationships_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_cni_relationships_maingraph.columns[0]: 'source', event_cni_relationships_maingraph.columns[1]: 'target'})
    df2=event_event_location_relationships_maingraph[event_event_location_relationships_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={event_event_location_relationships_maingraph.columns[0]: 'source', event_event_location_relationships_maingraph.columns[1]: 'target'})
    df3=evenet_vehicle_relationships_COP_maingraph[evenet_vehicle_relationships_COP_maingraph[':START_ID(Event)'].astype(str).isin(node_ids)].iloc[:, :2].rename(columns={evenet_vehicle_relationships_COP_maingraph.columns[0]: 'source', evenet_vehicle_relationships_COP_maingraph.columns[1]: 'target'})
    result_information_enqury = pd.concat([df1, df2,df3], axis=0)


    filtered_dict=pd.concat([result_event,result_location,result_party,result_vehicle,result_information_enqury],axis=0)
    filtered_dict.drop_duplicates(inplace=True)
    filtered_dict=filtered_dict.astype(str)

    # non_zero_values = filtered_dict[['source', 'target']].values[filtered_dict[['source', 'source']] != 0]
    # node_ids = non_zero_values.tolist()
    # node_ids=list(set(node_ids))

    node_ids = list(set(df['Record Key'].tolist()))
    node_ids = [item for item in node_ids if item != '0']


    #Gx = nx.from_pandas_edgelist(filtered_dict, 'source', 'target')
    #Gx.add_edges_from(filtered_dict)
    list_of_tuples = list(filtered_dict.itertuples(index=False, name=None))
    Gx = nx.Graph()
    Gx.add_edges_from(list_of_tuples)

    Dict_netwrok={}
    cluster=1
    for z in node_ids:
        try:
            shortest_paths = nx.single_source_shortest_path_length(Gx, source=z, cutoff=4)
            for x in [node for node, distance in shortest_paths.items()]:
                if x in Dict_netwrok:
                    Dict_netwrok[x].append(cluster)
                else:
                    Dict_netwrok[x]=[cluster]
            cluster=cluster+1
        except:
            pass


    def merge_values(lst):
        try:
            return ', '.join(map(str, lst))
        except:
            return ''

    import csv
    Dict_netwrok={key:Dict_netwrok.get(key, None) for key in node_ids}
    print('almost finsh with group generation')
    with open('multi_cluster_per_node.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in Dict_netwrok.items():
            try:
                writer.writerow([key, merge_values(value)])
            except:
                pass

    df2=pd.read_csv('multi_cluster_per_node.csv',names=['Record Key', 'dict_value'],header=None,encoding='ISO-8859-1')
    merged_df = df.reset_index().merge(df2, on='Record Key', how='left').set_index('index')
    print('merge done')


    import gc
    del df
    # Force garbage collection
    gc.collect()
    merged_df.dropna(inplace=True,subset = ['dict_value'])
    merged_df=merged_df[merged_df['Trans Type']!='Z']
    merged_df.reset_index(drop=True, inplace=True)
    merged_df['group']=0
    merged_df['group'] = compare_and_assign_group(merged_df, 'dict_value')

    #################################
    #################################This line will filter all the rego for admin peopel and I have to comment this , in he fture I might activate it again
    #################################I am not planning to use this for admin people so I always filter very admin in my code
    merged_df=merged_df[merged_df['Reg No'].astype(str).str.len()<=6]



    #counts = merged_df.groupby('group')['Record Key'].nunique()

    # Filter out groups where the count is equal to 1
    #groups_to_remove = counts[counts == 1].index

    # Remove rows corresponding to those groups from the original DataFrame
    #merged_df_filtered_df  = merged_df[~merged_df['group'].isin(groups_to_remove)].copy()
    #merged_df_filtered_df['group'] = compare_and_assign_group(merged_df_filtered_df, 'dict_value')
    df=merged_df.copy()
    df.drop(columns=['dict_value'],inplace=True)

    return df


def officer_event(df, off_event_relationships):
    merged = df.merge(off_event_relationships, left_on=['Reg No', 'Record Key'], right_on=[':START_ID(officer)', ':END_ID(Event)'], how='left')
    merged['record_exists'] = (merged['orderDate:DATE'] <= merged['Date']).astype(int)
    return merged.groupby(['Reg No', 'Date', 'Record Key'])['record_exists'].max().reset_index()

def officer_cni(df, off_event_relationships,event_cni_relationships):
    merged=df.merge(event_cni_relationships,left_on=df['Record Key'].astype(int), right_on=[':END_ID(CNI)'], how='left')
    merged=merged[merged['orderDate:DATE'] <= merged['Date']]
    merged.drop(columns=['orderDate:DATE'],inplace=True)
    merged = merged.merge(off_event_relationships, left_on=[':START_ID(Event)'], right_on=[':END_ID(Event)'], how='left')
    merged['record_exists'] = ((merged['orderDate:DATE'] <= merged['Date']) & (merged[':START_ID(officer)']==merged['Reg No'])).astype(int)
    return merged.groupby(['Reg No', 'Date', 'Record Key'])['record_exists'].max().reset_index()

def officer_cni_direct(df, off_cni_relationships):
    
    merged = df.merge(off_cni_relationships, left_on=['Reg No', df['Record Key'].astype(int)], right_on=[':START_ID(officer)', ':END_ID(CNI)'], how='left')
    merged['record_exists'] = (merged['orderDate:DATE'] <= merged['Date']).astype(int)
    return merged.groupby(['Reg No', 'Date', 'Record Key'])['record_exists'].max().reset_index()

def record_exists_invparty(df):

    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_INV_PARTY)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_invparty_event'}, inplace=True)
    df['record_exists_invparty_event'] = df['record_exists_invparty_event'].fillna(0)
    
    
    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni(dff, off_event_relationships_INV_PARTY,event_cni_relationships_INV_PARTY)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_invparty_cni'}, inplace=True)
    df['record_exists_invparty_cni'] = df['record_exists_invparty_cni'].fillna(0)

    print('finish with invparty')
    return df
###################################################
###################################################
###################################################
###################################################

def record_exists_charge(df):


    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_CHARGE)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_charge_event'}, inplace=True)
    df['record_exists_charge_event'] = df['record_exists_charge_event'].fillna(0)
    #df['record_exists_cop_party_link_event'] = df.apply(lambda row: query_neo4j(row['Reg No'], row['Date'],row['Record Key']), axis=1)


    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni(dff, off_event_relationships_CHARGE,event_cni_relationships_CHARGE)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_charge_cni'}, inplace=True)
    df['record_exists_charge_cni'] = df['record_exists_charge_cni'].fillna(0)



    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni_direct(dff, off_cni_relationships_CHARGE)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_charge_cni_direct'}, inplace=True)
    df['record_exists_charge_cni_direct'] = df['record_exists_charge_cni_direct'].fillna(0)
    print('finish with charge')
    return df

###################################################
###################################################maingraph officer, time consuming with zero value added to the system
###################################################
###################################################

# import pandas as pd
# from py2neo import Graph
# from datetime import datetime

# graph = Graph("bolt://127.0.0.1:7687",name="maingraphofficer", auth=("neo4j", "123456789"))


# # Apply the function to each row in the DataFrame
# dff=df[(df['Object Type']=='EVENT ENQUIRY')]
# dff=dff[['Reg No','Date','Record Key']]
# dff=dff.drop_duplicates()
# with ThreadPoolExecutor() as executor:
#     results = list(executor.map(check_record_exists_query_neo4j, [graph] * len(dff),[row for _, row in dff.iterrows()]))
# dff['record_exists_maingraphofficer_event'] = results
# df=df.merge(dff,left_on=['Reg No','Date','Record Key'],right_on=['Reg No','Date','Record Key'],how='left')
# df['record_exists_maingraphofficer_event'] = df['record_exists_maingraphofficer_event'].fillna(0)
# #df['record_exists_cop_party_link_event'] = df.apply(lambda row: query_neo4j(row['Reg No'], row['Date'],row['Record Key']), axis=1)


# dff=df[(df['Object Type']=='PARTY ENQUIRY')]
# dff=dff[['Reg No','Date','Record Key']]
# dff=dff.drop_duplicates()
# with ThreadPoolExecutor() as executor:
#     results = list(executor.map(check_record_exists_query_neo4j_2, [graph] * len(dff),[row for _, row in dff.iterrows()]))
# dff['record_exists_maingraphofficer_cni'] = results
# df=df.merge(dff,left_on=['Reg No','Date','Record Key'],right_on=['Reg No','Date','Record Key'],how='left')
# df['record_exists_maingraphofficer_cni'] = df['record_exists_maingraphofficer_cni'].fillna(0)


# dff=df[(df['Object Type']=='LOCATION ENQUIRY')]
# dff=dff[['Reg No','Date','Record Key']]
# dff=dff.drop_duplicates()
# with ThreadPoolExecutor() as executor:
#     results = list(executor.map(check_record_exists_query_neo4j_3_2, [graph] * len(dff),[row for _, row in dff.iterrows()]))
# dff['record_exists_maingraphofficer_party_location'] = results
# df=df.merge(dff,left_on=['Reg No','Date','Record Key'],right_on=['Reg No','Date','Record Key'],how='left')
# df['record_exists_maingraphofficer_party_location'] = df['record_exists_maingraphofficer_party_location'].fillna(0)
# #df['record_exists_cop_party_link_event'] = df.apply(lambda row: query_neo4j(row['Reg No'], row['Date'],row['Record Key']), axis=1)




# dff=df[(df['Object Type']=='LOCATION ENQUIRY')]
# dff=dff[['Reg No','Date','Record Key']]
# dff=dff.drop_duplicates()
# with ThreadPoolExecutor() as executor:
#     results = list(executor.map(check_record_exists_query_neo4j_3_1, [graph] * len(dff),[row for _, row in dff.iterrows()]))
# dff['record_exists_maingraphofficer_event_location'] = results
# df=df.merge(dff,left_on=['Reg No','Date','Record Key'],right_on=['Reg No','Date','Record Key'],how='left')
# df['record_exists_maingraphofficer_event_location'] = df['record_exists_maingraphofficer_event_location'].fillna(0)
# #df['record_exists_cop_party_link_event'] = df.apply(lambda row: query_neo4j(row['Reg No'], row['Date'],row['Record Key']), axis=1)

# dff=df[(df['Object Type']=='VEHICLE ENQUIRY')]
# dff=dff[['Reg No','Date','Record Key']]
# dff=dff.drop_duplicates()
# with ThreadPoolExecutor() as executor:
#     results = list(executor.map(check_record_exists_query_neo4j_4, [graph] * len(dff),[row for _, row in dff.iterrows()]))
# dff['record_exists_maingraphofficer_vehicle'] = results
# df=df.merge(dff,left_on=['Reg No','Date','Record Key'],right_on=['Reg No','Date','Record Key'],how='left')
# df['record_exists_maingraphofficer_vehicle'] = df['record_exists_maingraphofficer_vehicle'].fillna(0)

# print('finsih wih maingraphofficer')

###################################################
###################################################
###################################################
###################################################

import pandas as pd
from py2neo import Graph
from datetime import datetime
def record_exists_cop_party(df):

    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_PARTY_LINK)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_cop_party_link_event'}, inplace=True)
    df['record_exists_cop_party_link_event'] = df['record_exists_cop_party_link_event'].fillna(0)
    
    
    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni(dff, off_event_relationships_PARTY_LINK,event_cni_relationships_PARTY_LINK)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_cop_party_link_cni'}, inplace=True)
    df['record_exists_cop_party_link_cni'] = df['record_exists_cop_party_link_cni'].fillna(0)

    print('finsih wih coppartylink')
    return df



def record_exists_admin(df):


    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_ADMIN_LOG)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_admin_log'}, inplace=True)
    df['record_exists_admin_log'] = df['record_exists_admin_log'].fillna(0)
    print('almost finsh with adminlog')
    return df


def record_exists_event(df):

    
    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_EVENT)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_event'}, inplace=True)
    df['record_exists_event'] = df['record_exists_event'].fillna(0)
    print('finish with event')
    return df


def record_exists_cop_action(df):


    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_ACTION)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_cop_action_event'}, inplace=True)
    df['record_exists_cop_action_event'] = df['record_exists_cop_action_event'].fillna(0)
    
    
    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni_direct(dff, off_cni_relationships_ACTION)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_cop_action_cni'}, inplace=True)
    df['record_exists_cop_action_cni'] = df['record_exists_cop_action_cni'].fillna(0)
    
    print('finish with copaction')
    return df

def record_exists_copparty(df):

    
    dff=df[(df['Object Type']=='PARTY ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_cni_direct(dff, off_cni_relationships_COP_PARTY)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_copparty'}, inplace=True)
    df['record_exists_copparty'] = df['record_exists_copparty'].fillna(0)
    print('finish with copparty')
    return df


def record_exists_narrative(df):

    
    
    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[['Reg No','Date','Record Key']]
    dff=dff.drop_duplicates()
    results = officer_event(dff, off_event_relationships_NARRATIVE)
    df = df.merge(results, on=['Reg No', 'Date', 'Record Key'], how='left')
    df.rename(columns={'record_exists': 'record_exists_narrative'}, inplace=True)
    df['record_exists_narrative'] = df['record_exists_narrative'].fillna(0)

    print('finish with narrative')
    return df


def record_exists_cad(df):
    graph = Graph("bolt://127.0.0.1:7687",name="cad", auth=("neo4j", "123456789"))

    def query_neo4j(officer_ids):
        query = """
        WITH $officer_ids AS xIds
        MATCH (x:officer)-[:RELATED]->(y:callsign)-[:RELATED]->(z:event)
        WHERE x.officerId IN xIds
        RETURN DISTINCT x.officerId, z.eventId
        """
        result = graph.run(query, officer_ids=officer_ids)
        return [(record["x.officerId"], record["z.eventId"]) for record in result]

    # Function to process data in batches
    def process_in_batches(df, batch_size=5):
        officer_ids = df['Reg No'].astype(str).drop_duplicates().to_list()
        all_related_z = []

        for i in range(0, len(officer_ids), batch_size):
            batch_ids = officer_ids[i:i + batch_size]
            related_z = query_neo4j(batch_ids)
            all_related_z.extend(related_z)

        return all_related_z

    # Apply the function to the DataFrame
    related_z = process_in_batches(df)
    dff = pd.DataFrame(related_z, columns=['Reg No', 'Record Key'])
    dff['record_exists_cad'] = 1
    dff['Reg No']=dff['Reg No'].astype(int)
    df = df.merge(dff, on=['Reg No', 'Record Key'], how='left')
    df['record_exists_cad'] = df['record_exists_cad'].fillna(0)
    print('finish with cad')
    return df

def event_report_date(df):
    import pyodbc
    import pandas as pd

    # Set up connection parameters
    server = 'PPWDSQS03PI'
    database = 'COPS'
    trusted_connection = 'yes'  # Use Windows Authentication
    driver = '{SQL Server}'

    # Create a connection string
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

    # Establish a connection
    connection = pyodbc.connect(connection_string)

    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[dff['Record Key']!=0]
    dff=dff[['Record Key']]
    dff=dff.drop_duplicates()

    event_ref_nums = [str(val)[1:] for val in dff['Record Key']]

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]


    EVENT_REPORTED_DATE_TIME= pd.DataFrame()
    for chunk in chunk_list(event_ref_nums, 100):
        event_ref_nums_str = ','.join(chunk)
        query = f"""select EVENT_REF_NUM,EVENT_REPORTED_DATE_TIME from COPS.dbo.COP_EVENT
        where EVENT_REF_NUM IN ({event_ref_nums_str})"""


        chunk_result = pd.read_sql_query(query, connection)
        # Append the chunk results to the merged DataFrame
        EVENT_REPORTED_DATE_TIME = pd.concat([EVENT_REPORTED_DATE_TIME, chunk_result], ignore_index=True)

    # Execute the query
    EVENT_REPORTED_DATE_TIME['EVENT_REF_NUM']='E'+EVENT_REPORTED_DATE_TIME['EVENT_REF_NUM'].astype(int).astype(str)
    df=df.merge(EVENT_REPORTED_DATE_TIME,left_on='Record Key',right_on='EVENT_REF_NUM',how='left')
    print('finish with event report date')
    connection.close()
    return df

def date_mentioned_event_narra(df):
    #event mentioned in the narrative
    import pyodbc
    import pandas as pd
    import spacy

    # Set up connection parameters
    server = 'PPWDSQS03PI'
    database = 'i2IMS'
    trusted_connection = 'yes'  # Use Windows Authentication
    driver = '{SQL Server}'

    # Create a connection string
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

    # Establish a connection
    connection = pyodbc.connect(connection_string)

    dff=df[(df['Object Type']=='EVENT ENQUIRY')]
    dff=dff[dff['Record Key']!=0]
    dff=dff[['Record Key']]
    dff=dff.drop_duplicates()

    event_ref_nums = [str(val)[1:] for val in dff['Record Key']]

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    EVENT_DATE_MENTIONED = pd.DataFrame()

    for chunk in chunk_list(event_ref_nums, 100):
        event_ref_nums_str = ','.join(chunk)
        query = f"""
        select EventID, cast(Created as date) as dates, Narrative 
        from i2IMS.dbo.mm_EventToNarrative
        where EventID IN ({event_ref_nums_str})
        """
        
        ##########################
        ##########################
        ##########################TEMpraray chnages 

        # query = f"""
        # select dbo.Narrative_ToEvent.RightKey AS EventID, cast(dbo.Narrative.T as date) AS dates, REPLACE(REPLACE(REPLACE(dbo.Narrative.Narrative, CHAR(13), ' '), CHAR(10), ' '), CHAR(9), ' ') AS Narrative FROM dbo.Narrative INNER JOIN
        # dbo.Narrative_ToEvent ON dbo.Narrative.K = dbo.Narrative_ToEvent.LeftKey 
		# where dbo.Narrative_ToEvent.RightKey in({event_ref_nums_str})
        # """



        # Execute the query and fetch the results
        chunk_result = pd.read_sql_query(query, connection)
        
        # Append the chunk results to the merged DataFrame
        EVENT_DATE_MENTIONED = pd.concat([EVENT_DATE_MENTIONED, chunk_result], ignore_index=True)

    # Close the database connection
    connection.close()

    nlp = spacy.load("en_core_web_sm")
    date_labels = ['DATE']

    def extract_dates_spacy(text):
        doc = nlp(text)
        results = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in date_labels]
        return results

    def extract_dates_regex(text):
        import re
        pattern1 = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}(?:st|nd|rd|th)?,? \d{2,4}|\b(?:Sun(?:day)?|Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?)\b(?:day)?,? (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}(?:st|nd|rd|th)?,? \d{2,4}|\d{1,2}(?:st|nd|rd|th)? (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{2,4}|\b(?:Sun(?:day)?|Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?)\b(?:day)?,? (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}(?:st|nd|rd|th)?)\b'

        # Define the second regular expression pattern
        pattern2 = r'(?:\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})|(?:\b\d{2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b\s+\d{4})'

        pattern3=r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'


        pattern4 = r"\d{2}[/-]\d{2}[/-]\d{4}"
        pattern5 = r"\d{2}[/-]\d{2}[/-]\d{4}"
        pattern6= r"\d{4}-\d{2}-\d{2}"

        date_lists=[re.findall(regex, str(text)) for regex in [pattern1, pattern2,pattern4,pattern5,pattern6]]
        non_empty_lists = [tuple(date_list) for date_list in date_lists if date_list]

        # Remove duplicates by converting the list of tuples to a set and back to a list
        non_duplicate_lists = list(set(non_empty_lists))

        # Convert the tuples back to lists
        result = [list(date_list) for date_list in non_duplicate_lists]
        from dateutil.parser import parse

        formatted_dates = []
        for date_list in result:
            formatted_date_list = []
            for date_string in date_list:
                try:
                    parsed_date = parse(date_string)
                    formatted_date = parsed_date.strftime("%Y-%m-%d")
                    formatted_date_list.append(formatted_date)
                except:
                    continue
            formatted_dates.append(formatted_date_list)
        return formatted_dates

    # EVENT_DATE_MENTIONED['extracted_dates_spacy'] = EVENT_DATE_MENTIONED['Narrative'].apply(extract_dates_spacy)
    EVENT_DATE_MENTIONED['extracted_dates_regular'] = EVENT_DATE_MENTIONED['Narrative'].apply(extract_dates_regex)

    EVENT_DATE_MENTIONED['EventID']=EVENT_DATE_MENTIONED['EventID'].astype(int)
    EVENT_DATE_MENTIONED['EventID']=EVENT_DATE_MENTIONED['EventID'].astype(str)

    df_exploded = EVENT_DATE_MENTIONED.explode('extracted_dates_regular')

    # Merge with original dataframe on column A
    #merged_df = pd.merge(EVENT_DATE_MENTIONED[['EventID', 'dates']], df_exploded, on='EventID', how='right')

    df_exploded.to_csv('hidden.csv')
    merged_df=pd.read_csv('hidden.csv')

    from ast import literal_eval
    def aggregate_B_C(group):
        merged_B = group['dates']  # Merge column B
        merged_B=merged_B.unique().tolist()
        try:
            concatenated_C=[z for x in group['extracted_dates_regular'].dropna().apply(literal_eval) for z in x]
            unique_C = list(set(concatenated_C))  # Remove duplicates
        except:
            unique_C=[]
        if merged_B:
            unique_C.extend(merged_B)  # Append merged_B to unique_C
            unique_C = list(set(unique_C))  # Remove duplicates again, in case merged_B was already in concatenated_C
        return unique_C


    # Group by column A and apply custom aggregation function
    grouped_df = merged_df.groupby('EventID').apply(aggregate_B_C).reset_index()
    grouped_df.columns=['EventID','Date_Mentioned']


    grouped_df['EventID']='E'+grouped_df['EventID'].astype(int).astype(str)
    df=df.merge(grouped_df,left_on='Record Key',right_on='EventID',how='left')

    def set_S(row):
        w=pd.isna(row['Date_Mentioned'])
        if type(w)==bool:
            if w==False:
                if (row['Date'] in row['Date_Mentioned']):
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            if w.all()==False:
                #if (row['Date'].date().strftime('%Y-%m-%d') in row['Date_Mentioned']):
                if (row['Date'].strftime('%Y-%m-%d') in row['Date_Mentioned']):
                    return 1
                else:
                    return 0
            else:
                return 0

    # Apply the function to each row and assign the result to column S
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df['date_mentioned_event_narra'] = df.apply(set_S, axis=1)
    print('finish with date_mentioned_event_narra')
    return df


def cni_to_event_to_adminlog(df):

    def query_neo4j(tuple_list):
        tuple_list=tuple_list.tolist()
        result=event_cni_relationships_INV_PARTY[event_cni_relationships_INV_PARTY[':END_ID(CNI)'].astype(str).isin(tuple_list)][[':END_ID(CNI)',':START_ID(Event)']]
        return result


    def query_neo4j_2(event_ref_num,officer_id):
        event_ref_num=event_ref_num.tolist()
        result2=off_event_relationships_ADMIN_LOG[off_event_relationships_ADMIN_LOG[':END_ID(Event)'].isin(event_ref_num)]
        result2=result2[result2[':START_ID(officer)']==officer_id][[':END_ID(Event)','orderDate:DATE']]
        # query = f"""MATCH (parent)-[r:RELATED]->(child)
        # WHERE child.eventId in {event_ref_num} and parent.officerId=$officer_id
        # return child.eventId,r.orderDate"""
        # result2 = graph2.run(query,event_ref_num=event_ref_num,officer_id=officer_id).data()
        return result2


    try:
        dffw=df[(df['Object Type']=='PARTY ENQUIRY') & (df['Program']!='ICP897N')]
        dffw=dffw[['Date','Reg No','Object Type','Record Key']]
        dffw=dffw.drop_duplicates()
        related_event=query_neo4j(dffw['Record Key'])
        df2 =pd.DataFrame.from_dict(related_event)
        df2.columns=['Record Key', 'event_ref_num']
        dffw=dffw.merge(df2,left_on=dffw['Record Key'].astype(int),right_on=df2['Record Key'])
        related_event=query_neo4j_2(dffw['event_ref_num'],df['Reg No'].drop_duplicates().values[0]) #DANGER DANGER use officer reg could cause problem
        df2 =pd.DataFrame.from_dict(related_event)
        df2.columns=['event_ref_num','date']
        dffw=dffw.merge(df2)
        dffw['date'] = pd.to_datetime(dffw['date'],format='%Y-%m-%d')
        dffw['Date'] = pd.to_datetime(dffw['Date'],format='%Y-%m-%d')
        dffw=dffw[dffw['date'] <= dffw['Date']]
        #dffw = dffw.groupby(['0_x', '9']).apply(lambda x: x.assign(D=1)).reset_index(drop=True)
        dffw=dffw.groupby(['Date', 'Record Key_x']).size().reset_index(name='cni_related_offi_previous_event')
        dffw['cni_related_offi_previous_event']=1
    except:
        dffw=df[['Date','Record Key']]
        dffw=dffw.drop_duplicates()
        dffw['Date'] = pd.to_datetime(dffw['Date'],format='%Y-%m-%d')
        dffw['cni_related_offi_previous_event']=0
    return dffw


# def cni_related_offi_previous_event(df):
#     # Empty DataFrame to hold the final result
#     final_result = pd.DataFrame()

#     # Group by 'cluster' and apply the custom function
#     grouped = df.groupby('Reg No')

#     for name, group in grouped:
#         cni_related_offi_previous_event = cni_to_event_to_adminlog(group)
#         group['Date'] = pd.to_datetime(group['Date'],format='%Y-%m-%d')
#         merged_group=group.merge(cni_related_offi_previous_event,left_on=['Date','Record Key'],right_on=['Date','Record Key_x'],how='left')
#         merged_group['cni_related_offi_previous_event'] = merged_group['cni_related_offi_previous_event'].fillna(0)
#         final_result = pd.concat([final_result, merged_group], ignore_index=True)
#     df=final_result.copy()

#     #we dont have any kronus before 2020
#     df=df[df['Date']>='2020-01-01']
#     print('finish with cni related offi previous event')
#     return df

def process_group_pre(group):
    cni_related_offi_previous_event = cni_to_event_to_adminlog(group)
    group['Date'] = pd.to_datetime(group['Date'], format='%Y-%m-%d')
    merged_group = group.merge(cni_related_offi_previous_event, left_on=['Date', 'Record Key'], right_on=['Date', 'Record Key_x'], how='left')
    merged_group['cni_related_offi_previous_event'] = merged_group['cni_related_offi_previous_event'].fillna(0)
    return merged_group

def cni_related_offi_previous_event(df):
    # Group by 'Reg No'
    grouped = df.groupby('Reg No')

    # Use ThreadPoolExecutor to parallelize the processing of each group
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_group_pre, [group for name, group in grouped]))

    # Concatenate all results into a single DataFrame
    final_result = pd.concat(results, ignore_index=True)

    # Filter out dates before 2020
    final_result = final_result[final_result['Date'] >= '2020-01-01']
    
    print('Finished processing cni related offi previous event')
    return final_result



import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the Earth's surface given their latitude and longitude
    coordinates in degrees.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_of_earth_km = 6371  # Radius of the Earth in kilometers
    distance_km = radius_of_earth_km * c
    
    return distance_km

print("Generate the master location starts")
server = 'PPWDSQS03PI'
database = 'COPS'
trusted_connection = 'yes'  # Use Windows Authentication
driver = '{SQL Server}'

# Create a connection string
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

# Establish a connection
connection = pyodbc.connect(connection_string)
query = f"""select cast(LOCATION_REF_NUM as bigint) as LOCATION_REF_NUM,-LATITUDE_GEOCODE as LATITUDE_GEOCODE,LONGITUDE_GEOCODE from COPS.dbo.COP_LOCATION"""


    # Execute the query
locations_location_all = pd.read_sql(query, connection)
connection.close()
locations_location_all['LOCATION_REF_NUM'] = locations_location_all['LOCATION_REF_NUM'].astype(str)
print("Generate the master location ends")

def location_processing(df):
    import time
    start_time = time.time()
    from py2neo import Graph
    import pandas as pd
    import pyodbc
    dff=df.copy()
    graph = Graph("bolt://127.0.0.1:7687",name="maingraph", auth=("neo4j", "123456789"))
    dfff=dff[dff['Object Type'].isin(['PARTY ENQUIRY','LEGAL PROCESS ENQUIRY','CRIMINAL RECORD ENQUIRY','CRIMINAL HISTORY REPORT'])]
    node_ids = dfff['Record Key'].tolist()
    node_ids = list(set(node_ids))
    # Execute the Cypher query

    query = f"""
    MATCH (x)-[r:RELATED]->(party_location)
    WHERE x.cniId in {node_ids}
    RETURN x.cniId,party_location.party_location
    """
    result_party = graph.run(query).data()
    
    result_party = [(d['x.cniId'],d['party_location.party_location']) for d in result_party]
    result_party_df=pd.DataFrame(result_party,columns=['cni','location'])
    result_party_df['location']=result_party_df['location'].astype('int64')
    result_party_df=result_party_df.merge(dfff,left_on='cni',right_on='Record Key')[['cni','location','Date']]
    result_party_df.drop_duplicates(inplace=True)
    result_party_df['Date'] = pd.to_datetime(result_party_df['Date'])

    from py2neo import Graph
    import pandas as pd
    dfff=dff[dff['Object Type'].isin(['EVENT ENQUIRY'])]
    node_ids = dfff['Record Key'].tolist()
    node_ids = list(set(node_ids))
    # Execute the Cypher query
    query = f"""
    MATCH (x)-[r:RELATED]->(event_location:event_location)
    WHERE x.eventId in {node_ids}
    RETURN x.eventId,event_location.event_location
    """
    result_event = graph.run(query).data()
    result_event = [(d['x.eventId'],d['event_location.event_location']) for d in result_event]
    result_event_df=pd.DataFrame(result_event,columns=['event','location'])
    result_event_df['location']=result_event_df['location'].astype('int64')
    result_event_df=result_event_df.merge(dfff,left_on='event',right_on='Record Key')[['event','location','Date']]
    result_event_df.drop_duplicates(inplace=True)
    result_event_df['Date'] = pd.to_datetime(result_event_df['Date'])

    query = f"""
    MATCH (x:Event)-[]->(cni:CNI)-[]->(C:party_location)
    WHERE x.eventId in {node_ids}
    RETURN x.eventId,C.party_location
    """
    result_event = graph.run(query).data()
    result_event = [(d['x.eventId'],d['C.party_location']) for d in result_event]
    result_event_df_2=pd.DataFrame(result_event,columns=['event','location'])
    result_event_df_2['location']=result_event_df_2['location'].astype('int64')
    result_event_df_2=result_event_df_2.merge(dfff,left_on='event',right_on='Record Key')[['event','location','Date']]
    result_event_df_2.drop_duplicates(inplace=True)
    result_event_df_2['Date'] = pd.to_datetime(result_event_df_2['Date'])
    result_event_df = pd.concat([result_event_df, result_event_df_2], ignore_index=True)

    locations_location=dff[dff['Object Type']=='LOCATION ENQUIRY'][['Record Key','Date']]
    locations_location['Record Key']=locations_location['Record Key'].astype('int64')


    X1=[str(val) for val in locations_location['Record Key']]
    X2=[str(val) for val in result_party_df['location']]
    X3=[str(val) for val in result_event_df['location']]
    X1.extend(X2)
    X1.extend(X3)
    location_ref_nums_str = ','.join(X1)
    locations_location_temp = locations_location_all[locations_location_all['LOCATION_REF_NUM'].isin(X1)]
    locations_location_temp['LOCATION_REF_NUM']=locations_location_temp['LOCATION_REF_NUM'].astype('int64')

    locations_location=locations_location.merge(locations_location_temp,left_on='Record Key',right_on='LOCATION_REF_NUM',how='left')
    locations_location.drop_duplicates(inplace=True)
    result_party_df=result_party_df.merge(locations_location_temp,left_on='location',right_on='LOCATION_REF_NUM',how='left')
    result_party_df.drop_duplicates(inplace=True)
    result_party_df=result_party_df[result_party_df['LONGITUDE_GEOCODE']>0] ##################new changes
    result_event_df=result_event_df.merge(locations_location_temp,left_on='location',right_on='LOCATION_REF_NUM',how='left')
    result_event_df.drop_duplicates(inplace=True)

    # Set up connection parameters
    server = 'edwdb-sql'
    database = 'EDW_hr'
    trusted_connection = 'yes'  # Use Windows Authentication
    driver = '{SQL Server}'

    # Create a connection string
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

    # Establish a connection
    connection = pyodbc.connect(connection_string)
    officer=int(dff['Reg No'].unique()[0])
    #dff['Date'] = pd.to_datetime(dff['Date'], errors='coerce')
    column_name='registered_number'

    #query = f"SELECT  LABORLEVELDSC7,LABORLEVELDSC6,LABORLEVELDSC5,LABORLEVELDSC4,LABORLEVELDSC3,LABORLEVELDSC2,LABORLEVELDSC1 FROM {table_name} WHERE {column_name} = ? group by  LABORLEVELDSC7,LABORLEVELDSC6,LABORLEVELDSC5,LABORLEVELDSC4,LABORLEVELDSC3,LABORLEVELDSC2,LABORLEVELDSC1"

    query = f"SELECT  CONVERT(DATE,dt_shift_start) as dt_shift_start,LABORLEVELDSC7,LABORLEVELDSC6,LABORLEVELDSC5,LABORLEVELDSC4,LABORLEVELDSC3,LABORLEVELDSC2,LABORLEVELDSC1 FROM [EDW_HR].[kronos].[kronos] WHERE {column_name} = ? group by  LABORLEVELDSC7,LABORLEVELDSC6,LABORLEVELDSC5,LABORLEVELDSC4,LABORLEVELDSC3,LABORLEVELDSC2,LABORLEVELDSC1,CONVERT(DATE,dt_shift_start)"

    # Load the query result into a pandas DataFrame
    officer_kronus_date = pd.read_sql(query, connection, params=(officer,))

    connection.close()

    officer_kronus_date.set_index('dt_shift_start', inplace=True)
    eliminate_list=['WORKSHOP',	'STRIKE FORCE',	'FORUM',	'SEARCH',	'OVERTIME',	'SPECIAL OPERATION',	'TRAFFIC SERVICES',	'SOCCER',	'TRAINING',	'CATCH TRAINING',	'CITY TO SURF',	'CRIME',	'EXECUTIVE OFFICER',	'INTERVIEW',	'OC',	'OPERATION SLOW DOWN',	'OPERATION VIKINGS',	'OPERATIONS',	'PHYSICAL',]
    officer_kronus_date = officer_kronus_date.applymap(lambda x: '{-}' if x.lower() in [item.lower() for item in eliminate_list] else x)
    # officer_kronus_date=officer_kronus_date.stack(level=0).reset_index(level=0)
    # officer_kronus_date.drop_duplicates(inplace=True)
    
    # combined_values = list(set(officer_kronus_date[0]))
    # filtered_values = [value for value in combined_values if "'" not in value and '"' not in value]
    # placeholders= ', '.join(f"'{unused}'" for unused in filtered_values)


    # # Set up connection parameters
    # server = 'PPWDSQS03PI'
    # database = 'COPS'
    # trusted_connection = 'yes'  # Use Windows Authentication
    # driver = '{SQL Server}'

    # # Create a connection string
    # connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'

    # # Establish a connection
    # connection = pyodbc.connect(connection_string)

    # query = f'SELECT PROPERTY_NAME,-LATITUDE_GEOCODE as LATITUDE_GEOCODE,LONGITUDE_GEOCODE FROM COPS.dbo.COP_LOCATION WHERE PROPERTY_NAME in ({placeholders}) and LATITUDE_GEOCODE is not null Group by LATITUDE_GEOCODE,LONGITUDE_GEOCODE,PROPERTY_NAME'

    # # Load the query result into a pandas DataFrame
    # officer_lat_long = pd.read_sql(query, connection)
    # connection.close()

    # officer_kronus_date=officer_kronus_date.merge(officer_lat_long,left_on=officer_kronus_date[0].str.lower(),right_on=officer_lat_long['PROPERTY_NAME'].str.lower())
    # officer_kronus_date=officer_kronus_date[['dt_shift_start','LATITUDE_GEOCODE','LONGITUDE_GEOCODE']].drop_duplicates()
    # officer_kronus_date.columns=['dt_shift_start','LATITUDE_GEOCODE_OFFICER','LONGITUDE_GEOCODE_OFFICER']
    def process_column(col):
        combined_values = col.unique()
        # filtered_values = [value for value in combined_values if "'" not in value and '"' not in value]
        # placeholders= ', '.join(f"'{unused}'" for unused in filtered_values)

        filtered_values = [value for value in combined_values if "'" not in value and '"' not in value]
        extended_values = filtered_values + [value.replace('Police Area Command', 'PAC') for value in filtered_values if 'Police Area Command' in value]
        placeholders= ', '.join(f"'{unused}'" for unused in extended_values)


        server = 'PPWDSQS03PI'
        database = 'COPS'
        trusted_connection = 'yes'  
        driver = '{SQL Server}'
        connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}'
        connection = pyodbc.connect(connection_string)
        # query =f"""with cte as (
        # select PROPERTY_NAME,LOCATION_REF_NUM,LATITUDE_GEOCODE,LONGITUDE_GEOCODE from cops.dbo.COP_LOCATION
        # where PROPERTY_NAME in ({placeholders})
        # group by PROPERTY_NAME,LOCATION_REF_NUM,LATITUDE_GEOCODE,LONGITUDE_GEOCODE)
        # select top 1  b.PROPERTY_NAME,a.LOCATION_REF_NUM,b.LATITUDE_GEOCODE,b.LONGITUDE_GEOCODE from cops.dbo.COP_INV_LOCATION a join cte b on a.LOCATION_REF_NUM=b.LOCATION_REF_NUM
        # group by b.PROPERTY_NAME,a.LOCATION_REF_NUM,b.LATITUDE_GEOCODE,b.LONGITUDE_GEOCODE
        # order by count(*) desc"""
        query =f"""with cte as (
        select PROPERTY_NAME,LOCATION_REF_NUM,LATITUDE_GEOCODE,LONGITUDE_GEOCODE from cops.dbo.COP_LOCATION
        where PROPERTY_NAME in ({placeholders})
        group by PROPERTY_NAME,LOCATION_REF_NUM,LATITUDE_GEOCODE,LONGITUDE_GEOCODE)
		,cte1 as
		(
		select b.PROPERTY_NAME,a.LOCATION_REF_NUM,b.LATITUDE_GEOCODE,b.LONGITUDE_GEOCODE,count(*) as [Value]  from cops.dbo.COP_INV_LOCATION a join cte b on a.LOCATION_REF_NUM=b.LOCATION_REF_NUM
        group by b.PROPERTY_NAME,a.LOCATION_REF_NUM,b.LATITUDE_GEOCODE,b.LONGITUDE_GEOCODE
		)
		,cte2 as
		(
		select b.PROPERTY_NAME,b.LOCATION_REF_NUM,b.LATITUDE_GEOCODE,b.LONGITUDE_GEOCODE,[Value],row_number() over(PARTITION BY  b.PROPERTY_NAME ORDER BY [Value] Desc) as [order] from cte1 b

		)
		select LTRIM(RTRIM(PROPERTY_NAME)) as PROPERTY_NAME,LOCATION_REF_NUM,LATITUDE_GEOCODE,LONGITUDE_GEOCODE from cte2 where [order]=1"""
        #query = """SELECT PROPERTY_NAME,-LATITUDE_GEOCODE as LATITUDE_GEOCODE,LONGITUDE_GEOCODE FROM COPS.dbo.COP_LOCATION WHERE PROPERTY_NAME in ({placeholders}) and LATITUDE_GEOCODE is not null Group by LATITUDE_GEOCODE,LONGITUDE_GEOCODE,PROPERTY_NAME"""
        officer_lat_long = pd.read_sql(query, connection)

           ######################Danger 

        filtered_df = officer_lat_long[officer_lat_long['PROPERTY_NAME'].str.lower().str.contains('pac')]
        new_rows = filtered_df.copy()
        new_rows['PROPERTY_NAME'] = new_rows['PROPERTY_NAME'].str.replace('PAC', 'Police Area Command')
        officer_lat_long = pd.concat([officer_lat_long, new_rows], ignore_index=True)

                      ######################Danger 

        connection.close()

        def create_tuple(row):
            return (row['PROPERTY_NAME'], -row['LATITUDE_GEOCODE'], row['LONGITUDE_GEOCODE'])

        temp_df=col.to_frame().merge(officer_lat_long,left_on=col.to_frame()[col.name].str.lower(),right_on=officer_lat_long['PROPERTY_NAME'].str.lower(),how='left').set_index(col.to_frame().index)

        return temp_df.apply(create_tuple, axis=1)

    #print(process_column(officer_kronus_date['LABORLEVELDSC7']))	

    # # Traverse each column and apply the function
    for column in officer_kronus_date.columns:
        officer_kronus_date[column]=process_column(officer_kronus_date[column])

    import numpy as np
    def select_leftmost_non_empty_tuple(row):
        for value in row:
            if any(pd.notna(element) for element in value):  # Check if any element in the tuple is not NaN
                return value
        return (np.nan, np.nan ,np.nan)  # Return a tuple of NaNs if all values are empty


    officer_kronus_date['LeftmostValue'] = officer_kronus_date.apply(select_leftmost_non_empty_tuple, axis=1)
    officer_kronus_date[['PROPERTY_NAME','LATITUDE_GEOCODE','LONGITUDE_GEOCODE']]=pd.DataFrame(officer_kronus_date['LeftmostValue'].tolist(), index=officer_kronus_date.index)
    officer_kronus_date=officer_kronus_date.reset_index()[['dt_shift_start','LATITUDE_GEOCODE','LONGITUDE_GEOCODE']].drop_duplicates()

    # officer_kronus_date=officer_kronus_date.merge(officer_lat_long,left_on=officer_kronus_date[0].str.lower(),right_on=officer_lat_long['PROPERTY_NAME'].str.lower())
    # officer_kronus_date=officer_kronus_date[['dt_shift_start','LATITUDE_GEOCODE','LONGITUDE_GEOCODE']].drop_duplicates()
    officer_kronus_date.columns=['dt_shift_start','LATITUDE_GEOCODE_OFFICER','LONGITUDE_GEOCODE_OFFICER']

    officer_kronus_date['dt_shift_start'] = pd.to_datetime(officer_kronus_date['dt_shift_start'])

    result_party_df=result_party_df.merge(officer_kronus_date,left_on='Date',right_on='dt_shift_start')
    result_event_df=result_event_df.merge(officer_kronus_date,left_on='Date',right_on='dt_shift_start')
    locations_location=locations_location.merge(officer_kronus_date,left_on='Date',right_on=officer_kronus_date['dt_shift_start'])
    
    try:
        result_party_df['distance_party'] = result_party_df.apply(lambda row: haversine(row['LATITUDE_GEOCODE_OFFICER'], row['LONGITUDE_GEOCODE_OFFICER'], row['LATITUDE_GEOCODE'], row['LONGITUDE_GEOCODE']), axis=1)
        result_party_df = result_party_df.groupby(['cni','Date'])['distance_party'].min().reset_index()
        result_party_df['cni']=result_party_df['cni'].astype(str)
    
        result_event_df['distance_event'] = result_event_df.apply(lambda row: haversine(row['LATITUDE_GEOCODE_OFFICER'], row['LONGITUDE_GEOCODE_OFFICER'], row['LATITUDE_GEOCODE'], row['LONGITUDE_GEOCODE']), axis=1)
        result_event_df = result_event_df.groupby(['event','Date'])['distance_event'].min().reset_index()
        result_event_df['event']=result_event_df['event'].astype(str)
    
        locations_location['distance_location'] = locations_location.apply(lambda row: haversine(row['LATITUDE_GEOCODE_OFFICER'], row['LONGITUDE_GEOCODE_OFFICER'], row['LATITUDE_GEOCODE'], row['LONGITUDE_GEOCODE']), axis=1)
        locations_location = locations_location.groupby(['Record Key','Date'])['distance_location'].min().reset_index()
        locations_location['Record Key']=locations_location['Record Key'].astype(str)
    except:
        result_party_df['distance_party'] = -1000
        result_party_df = result_party_df.groupby(['cni','Date'])['distance_party'].min().reset_index()
        result_party_df['cni']=result_party_df['cni'].astype(str)
    
        result_event_df['distance_event'] = -1000
        result_event_df = result_event_df.groupby(['event','Date'])['distance_event'].min().reset_index()
        result_event_df['event']=result_event_df['event'].astype(str)
    
        locations_location['distance_location'] = -1000
        locations_location = locations_location.groupby(['LOCATION_REF_NUM','Date'])['distance_location'].min().reset_index()
        locations_location['LOCATION_REF_NUM']=locations_location['LOCATION_REF_NUM'].astype(str)

    del graph
    gc.collect()
    return locations_location,result_event_df,result_party_df


def merge_dataframes(df, locations_location, result_event_df, result_party_df):
    merged_df = df.copy()
    #*****************************************************************************************************************************I filter this might un filter soon
    #merged_df = merged_df.merge(locations_location,left_on=['Record Key','Date'],right_on=['Record Key','Date'],how='left')
    #merged_df = merged_df.merge(result_event_df,left_on=['Record Key','Date'],right_on=['event','Date'],how='left')
    #****************************************************************************************************************************************************************
    merged_df = merged_df.merge(result_party_df,left_on=['Record Key','Date'],right_on=['cni','Date'],how='left')
    return merged_df


def process_group(name_group_tuple):
    name, group = name_group_tuple
    locations_location, result_event_df, result_party_df = location_processing(group)
    merged_group = merge_dataframes(group, locations_location, result_event_df, result_party_df)
    return merged_group



def loc_cni_event_distance(df):
    grouped = df.groupby('Reg No')
    groups = [(name, group) for name, group in grouped]

    final_result = pd.DataFrame()

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_group, groups))
    final_result = pd.concat(results, ignore_index=True)
    print('finish with location cni event distance')
    return final_result

def iterative_max_sum_update(df):
    while True:
        # Step 1: Group by Cluster and update nodes' sum with the maximum sum in the cluster
        cluster_max = df.groupby('group')['sum group'].max().reset_index()
        cluster_max.columns = ['group', 'cluster_max_sum']
        df = df.merge(cluster_max, on='group', how='left')
        df['new_sum'] = df[['sum group', 'cluster_max_sum']].max(axis=1)
        if (df['sum group'] == df['new_sum']).all():
            break
        df['sum group'] = df['new_sum']
        df.drop(columns=['cluster_max_sum', 'new_sum'], inplace=True)
        
        # Step 2: Group by Node and update clusters' sum with the maximum sum for each node
        node_max = df.groupby('Record Key')['sum group'].max().reset_index()
        node_max.columns = ['Record Key', 'node_max_sum']
        df = df.merge(node_max, on='Record Key', how='left')
        df['new_sum'] = df[['sum group', 'node_max_sum']].max(axis=1)
        if (df['sum group'] == df['new_sum']).all():
            break
        df['sum group'] = df['new_sum']
        df.drop(columns=['node_max_sum', 'new_sum'], inplace=True)
    #df.drop(columns=['node_max_sum', 'new_sum'], inplace=True)
    return df



def main():

    main_df=pd.read_csv('Program_Type_All_260_officer.csv')
    main_df['NewColumn'] = range(1, len(main_df) + 1)
    main_df = main_df.sort_values(by=['Reg No', 'NewColumn'])
    main_df.to_csv('Program_Type_All_260_officer.csv',index=False)


    main_df=pd.read_csv('Program_Type_All_260_officer.csv')
    main_df['Record Key'] = main_df['Record Key'].astype(str)
    main_df.loc[main_df['Object Type'] == "EVENT ENQUIRY", 'Record Key'] = 'E' + main_df['Record Key']
    main_df.loc[main_df['Object Type'] == "INFORMATION REPORT ENQUIRY", 'Record Key'] = 'I' + main_df['Record Key']

    unique_values = main_df['Reg No'].unique()

    batch_size = 20
    results = pd.DataFrame()
    for i in range(0, len(unique_values), batch_size):
        start_time = time.time()
        print ('process batch')
        batch_values = unique_values[i:i + batch_size]
        df = main_df[main_df['Reg No'].isin(batch_values)]
    
        df=cluster_creation_no_neo(df) #cluster_creation(df)
        print(len(df))
        df=record_exists_invparty(df)
        print(len(df))
        df=record_exists_charge(df)
        print(len(df))
        df=record_exists_cop_party(df)
        print(len(df))
        df=record_exists_admin(df)
        print(len(df))
        df=record_exists_event(df)
        print(len(df))
        df=record_exists_cop_action(df)
        print(len(df))
        df=record_exists_copparty(df)
        print(len(df))
        df=record_exists_cad(df)
        print(len(df))
        df=record_exists_narrative(df)
        print(len(df))
        df=event_report_date(df)
        print(len(df))
        df=date_mentioned_event_narra(df)
        print(len(df))
        df=cni_related_offi_previous_event(df)
        print(len(df))
        df=loc_cni_event_distance(df)
        print(len(df))
        ##results.append(df)
        end_time = time.time()
        # Calculate the elapsed time
        print("Tiem is :::::::::::::::::::::::::::")
        print(end_time - start_time)
        df['sum group']=df[['record_exists_invparty_event','record_exists_invparty_cni','record_exists_cop_party_link_event','record_exists_cop_party_link_cni','record_exists_admin_log','record_exists_event','record_exists_cop_action_event','record_exists_cop_action_cni','record_exists_copparty','record_exists_narrative','record_exists_cad','cni_related_offi_previous_event','date_mentioned_event_narra','record_exists_charge_event','record_exists_charge_cni','record_exists_charge_cni_direct']].sum(axis=1)
        df = iterative_max_sum_update(df)
        results = pd.concat([results, df], ignore_index=True)

    results.to_csv('Benchmarkv2_148_officers.csv')

    

if __name__ == '__main__':
    main()





# Perform the task
#final_result['sum group']=final_result[['record_exists_invparty_event','record_exists_invparty_cni','record_exists_cop_party_link_event','record_exists_cop_party_link_cni','record_exists_admin_log','record_exists_event','record_exists_cop_action_event','record_exists_cop_action_cni','record_exists_copparty','record_exists_narrative','record_exists_cad','cni_related_offi_previous_event','date_mentioned_event_narra','record_exists_maingraphofficer_event','record_exists_maingraphofficer_cni','record_exists_maingraphofficer_party_location','record_exists_maingraphofficer_event_location','record_exists_maingraphofficer_vehicle']].sum(axis=1)





#after finding the sum group, I filter those that the suim group is equal to zero, I also remove those officer that have cni iwth distance equal to -1000 since it will show that application afiled to find the lat anf long
#then serahc for only party enqury and CIX and idtsnace gerater than 15 , then for those left I do a fuzzy wuzzy to look athe possibity of  serahcing for a simailr names, and the cluster those simairl names *where fuzzy wuzzy is gerate than 80 
#and save teh fuianl results to a file



# a=final_result[final_result['distance_party']==-1000]['Reg No'].drop_duplicates()
# filtered_df = final_result[~final_result['Reg No'].isin(a)]
# filtered_df=filtered_df[filtered_df['sum group']==0]
# filtered_df=filtered_df[filtered_df['Object Type']=='PARTY ENQUIRY']
# filtered_df=filtered_df[filtered_df['Program'].str.contains("CIX", na=False)]
# filtered_df=filtered_df[filtered_df['distance_party']>15]
# filtered_df.to_csv('44423_large_bench_mark.csv')
# fuzzy_wizzy=filtered_df[['Record Key','Record Value']]
# fuzzy_wizzy.to_csv('fuzzy_wizzy1234.csv')
# fuzzy_wizzy.drop_duplicates(inplace=True)

# import pandas as pd
# from fuzzywuzzy import fuzz

# # Sample DataFrame (replace with your actual DataFrame)

# fuzzy_wizzy_df=fuzzy_wizzy[['Record Value']]

# # Function to calculate fuzzy similarity
# def calculate_similarity(df):
#     similarity_matrix = pd.DataFrame(index=df.index, columns=df.index)

#     for i in df.index:
#         for j in df.index:
#             similarity_matrix.at[i, j] = fuzz.ratio(df.at[i, 'Record Value'], df.at[j, 'Record Value'])

#     return similarity_matrix

# # Calculate the similarity matrix
# similarity_matrix = calculate_similarity(fuzzy_wizzy_df)


# from collections import defaultdict
# def find_clusters(similarity_matrix, threshold):
#     clusters = defaultdict(list)
#     cluster_id = 0
#     visited = set()

#     for i in similarity_matrix.index:
#         if i not in visited:
#             cluster_id += 1
#             queue = [i]
#             while queue:
#                 idx = queue.pop(0)
#                 if idx not in visited:
#                     visited.add(idx)
#                     clusters[cluster_id].append(idx)
#                     for j in similarity_matrix.index:
#                         if j not in visited and similarity_matrix.at[idx, j] > threshold:
#                             queue.append(j)
#     return clusters


# clusters = find_clusters(similarity_matrix, 80)

# clustered_records = []

# for cluster_id, indices in clusters.items():
#     for index in indices:
#         clustered_records.append({
#             'Cluster ID': cluster_id,
#             'Record Index': index,
#             'Record Value': fuzzy_wizzy_df.at[index, 'Record Value']
#         })

# clustered_df = pd.DataFrame(clustered_records)

# # Save the clustered records to a CSV file
# clustered_df.to_csv('clustered_records.csv', index=False)

