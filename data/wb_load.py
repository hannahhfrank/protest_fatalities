import wbgapi as wb
import pandas as pd

# The World Bank data is loaded via the API
# But can be manually downloaded here: https://databank.worldbank.org/source/world-development-indicators

# Specify variables for call
feat_dev = ["NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "SP.POP.TOTL", # Population size
            ]

# Specify country codes, left GW and right WB
# Get WB country codes by downloading one variable manually: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
# GW codes: http://ksgleditsch.com/data-4.html
c_codes = {'Afghanistan': [700, 'AFG'],
            'Albania': [339, 'ALB'],
            'Algeria': [615, 'DZA'],
            'Andorra': [232, 'AND'], 
            'Angola': [540, 'AGO'], 
            'Antigua and Barbuda': [58, 'ATG'], 
            'Argentina': [160, 'ARG'], 
            'Armenia': [371, 'ARM'], 
            'Australia': [900, 'AUS'],
            'Austria': [305, 'AUT'], 
            'Azerbaijan': [373, 'AZE'],
            'Bahamas': [31, 'BHS'],
            'Bahrain': [692, 'BHR'],
            'Bangladesh': [771, 'BGD'],
            'Barbados': [53, 'BRB'],
            'Belarus': [370, 'BLR'],
            'Belgium': [211, 'BEL'],
            'Belize': [80, 'BLZ'],
            'Benin': [434, 'BEN'],
            'Bhutan': [760, 'BTN'],
            'Bolivia': [145, 'BOL'],
            'Bosnia and Herzegovina': [346, 'BIH'],
            'Botswana': [571, 'BWA'],
            'Brazil': [140, 'BRA'],
            'Brunei Darussalam': [835, 'BRN'],
            'Bulgaria': [355, 'BGR'],
            'Burkina Faso': [439, 'BFA'],
            'Burundi': [516, 'BDI'],
            'Cabo Verde': [402, 'CPV'],
            'Cambodia (Kampuchea)': [811, 'KHM'],
            'Cameroon': [471, 'CMR'],
            'Canada': [20, 'CAN'],
            'Central African Republic': [482, 'CAF'],
            'Chad': [483, 'TCD'],
            'Chile': [155, 'CHL'],
            'China': [710, 'CHN'],
            'Colombia': [100, 'COL'],
            'Comoros': [581, 'COM'],
            'Congo': [484, 'COG'],
            'Costa Rica': [94,'CRI'],
            'Croatia': [344, 'HRV'],
            'Cuba': [40, 'CUB'],
            'Cyprus': [352, 'CYP'],
            'Czechia': [316, 'CZE'],
            'Democratic Peoples Republic of Korea': [731, 'PRK'],
            'DR Congo (Zaire)': [490, 'COD'],
            'Denmark': [390, 'DNK'],
            'Djibouti': [522, 'DJI'],
            'Dominica': [54, 'DMA'],
            'Dominican Republic': [42, 'DOM'],
            'East Timor': [860, 'TLS'], 
            'Ecuador': [130, 'ECU'],
            'Egypt': [651, 'EGY'],
            'El Salvador': [92, 'SLV'],
            'Equatorial Guinea': [411, 'GNQ'],
            'Eritrea': [531, 'ERI'],
            'Estonia': [366, 'EST'],
            'eSwatini': [572, 'SWZ'],
            'Ethiopia': [530, 'ETH'],
            'Fiji': [950, 'FJI'],
            'Finland': [375, 'FIN'],
            'France': [220, 'FRA'],
            'Gabon': [481, 'GAB'],
            'Gambia': [420, 'GMB'],
            'Georgia': [372, 'GEO'],
            'Germany': [260, 'DEU'],
            'Ghana': [452, 'GHA'],
            'Greece': [350, 'GRC'],
            'Grenada': [55, 'GRD'],
            'Guatemala': [90, 'GTM'],
            'Guinea': [438, 'GIN'],
            'Guinea-Bissau': [404, 'GNB'],
            'Guyana': [110, 'GUY'],
            'Haiti': [41, 'HTI'],
            'Honduras': [91, 'HND'],
            'Hungary': [310, 'HUN'],
            'Iceland': [395, 'ISL'],
            'India': [750, 'IND'],
            'Indonesia': [850, 'IDN'],
            'Iran': [630, 'IRN'],
            'Iraq': [645, 'IRQ'],
            'Ireland': [205, 'IRL'],
            'Israel': [666, 'ISR'],
            'Italy': [325, 'ITA'],
            'Ivory Coast': [437, 'CIV'],
            'Jamaica': [51, 'JAM'],
            'Japan': [740, 'JPN'],
            'Jordan': [663, 'JOR'],
            'Kazakhstan': [705, 'KAZ'],
            'Kenya': [501, 'KEN'],
            'Kiribati': [970, 'KIR'],
            'Kuwait': [690, 'KWT'],
            'Kosovo': [347, 'XKX'],
            'Kyrgyzstan': [703, 'KGZ'],
            'Laos': [812, 'LAO'],
            'Latvia': [367, 'LVA'],
            'Lebanon': [660, 'LBN'],
            'Lesotho': [570, 'LSO'],
            'Liberia': [450, 'LBR'],
            'Libya': [620, 'LBY'],
            'Liechtenstein': [223, 'LIE'],
            'Lithuania': [368, 'LTU'],
            'Luxembourg': [212, 'LUX'],
            'Madagascar': [580, 'MDG'],
            'Malawi': [553, 'MWI'],
            'Malaysia': [820, 'MYS'],
            'Maldives': [781, 'MDV'],
            'Mali': [432, 'MLI'],
            'Malta': [338, 'MLT'],
            'Marshall Islands': [983, 'MHL'],
            'Mauritania': [435, 'MRT'],                  
            'Mauritius': [590, 'MUS'],   
            'Mexico': [70, 'MEX'],        
            'Micronesia (Federated States of)': [987, 'FSM'],        
            'Monaco': [221, 'MCO'], 
            'Mongolia': [712, 'MNG'],
            'Montenegro': [341, 'MNE'],
            'Morocco': [600, 'MAR'],
            'Mozambique': [541, 'MOZ'],
            'Myanmar (Burma)': [775, 'MMR'],
            'Namibia': [565, 'NAM'],
            'Nauru': [971, 'NRU'],
            'Nepal': [790, 'NPL'],
            'Netherlands': [210, 'NLD'],
            'New Zealand': [920, 'NZL'],
            'Nicaragua': [93, 'NIC'], 
            'Niger': [436, 'NER'],
            'Nigeria': [475, 'NGA'],
            'North Macedonia': [343, 'MKD'],
            'Norway': [385, 'NOR'],
            'Oman': [698, 'OMN'],
            'Pakistan': [770, 'PAK'],
            'Palau': [986, 'PLW'], 
            'Panama': [95, 'PAN'], 
            'Papua New Guinea': [910, 'PNG'],
            'Paraguay': [150, 'PRY'],
            'Peru': [135, 'PER'],
            'Philippines': [840, 'PHL'],
            'Poland': [290, 'POL'],
            'Portugal': [235, 'PRT'],
            'Qatar': [694, 'QAT'],
            'Republic of Moldova': [359, 'MDA'],
            'Romania': [360, 'ROU'],
            'Russia': [365, 'RUS'],
            'Rwanda': [517, 'RWA'],
            'Saint Kitts and Nevis': [60, 'KNA'],
            'Saint Lucia': [56, 'LCA'],
            'Saint Vincent and the Grenadines': [57, 'VCT'],
            'Samoa': [990, 'WSM'],
            'San Marino': [331, 'SMR'],
            'Sao Tome and Principe': [403, 'STP'],
            'Saudi Arabia': [670, 'SAU'],
            'Senegal': [433, 'SEN'],
            'Serbia': [340, 'SRB'],
            'Seychelles': [591, 'SYC'],
            'Sierra Leone': [451, 'SLE'],
            'Singapore': [830, 'SGP'],
            'Slovakia': [317, 'SVK'],
            'Slovenia': [349, 'SVN'],
            'Solomon Islands': [940, 'SLB'],
            'Somalia': [520, 'SOM'],
            'South Africa': [560, 'ZAF'],
            'South Korea': [732, 'KOR'],
            'South Sudan': [626, 'SSD'], 
            'Spain': [230, 'ESP'], 
            'Sri Lanka': [780, 'LKA'],
            'Sudan': [625, 'SDN'],
            'Suriname': [115, 'SUR'],
            'Sweden': [380, 'SWE'],
            'Switzerland': [225, 'CHE'],
            'Syria': [652, 'SYR'],
            'Tajikistan': [702, 'TJK'],
            'Tanzania': [510, 'TZA'],
            'Thailand': [800, 'THA'],
            'Taiwan': [713, 'XYZ'], # Not in WB
            'Togo': [461, 'TGO'],
            'Tonga': [972, 'TON'],
            'Trinidad and Tobago': [52, 'TTO'],
            'Tunisia': [616, 'TUN'],
            'Turkey': [640, 'TUR'],
            'Turkmenistan': [701, 'TKM'],
            'Tuvalu': [973, 'TUV'],
            'Uganda': [500, 'UGA'],
            'Ukraine': [369, 'UKR'],
            'United Arab Emirates': [696, 'ARE'],
            'United Kingdom': [200, 'GBR'],
            'United States': [2, 'USA'],
            'Uruguay': [165, 'URY'],
            'Uzbekistan': [704, 'UZB'],
            'Vanuatu': [935, 'VUT'],
            'Venezuela': [101, 'VEN'],
            'Vietnam': [816, 'VNM'],
            'Yemen (North Yemen)': [678, 'YEM'],
            'Zambia': [551, 'ZMB'],
            'Zimbabwe': [552, 'ZWE'],
            }

# Convert country codes from dictionary into df
c_codes = pd.DataFrame.from_dict(c_codes,orient='index')
c_codes = c_codes.reset_index()
c_codes.columns = ['country','gw_codes','iso_alpha3']

# Specify countries for call  
c_list=list(c_codes.iso_alpha3)
c_list=[char for char in c_list if char != "XYZ"] # Exclude Taiwan

# Specify years for call
years=list(range(1989, 2024, 1))

# Define out df
wdi = pd.DataFrame()

# Get data for each year and merge
for i in years:
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, c_list, [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  

# Add country and country codes: Merge GW codes over WB country codes 
wdi_final = pd.merge(wdi,c_codes[['gw_codes','iso_alpha3',"country"]],how='left',left_on=['economy'],right_on=['iso_alpha3'])

# Drop duplicates WB country codes
wdi_final = wdi_final.drop(columns=['economy'])

# Sort columns, so that year, country and country codes appear at beginning
wdi_final = wdi_final[['country','year','iso_alpha3','gw_codes'] + [c for c in wdi_final.columns if c not in ['country','year','iso_alpha3','gw_codes']]]

# Print head of df to confirm load   
print("Obtained data")
print(wdi_final.head())

# Sort and reset index
wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
wdi_final = wdi_final.reset_index(drop=True)

# Save data  
wdi_final.to_csv("wdi.csv") 
print(wdi_final.duplicated(subset=["year","gw_codes","country"]).any())
print(wdi_final.duplicated(subset=["year","country"]).any())
print(wdi_final.duplicated(subset=["year","gw_codes"]).any())
wdi_final.dtypes




