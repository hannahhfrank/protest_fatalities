### This file generates a csv and dictionary file containing UCDP GED data on series level ###

### Load libraries -------
import pandas as pd
from functions import simple_imp_grouped,linear_imp_grouped
import matplotlib.pyplot as plt
import wbgapi as wb

                                #############
                                ### ACLED ###
                                #############

### Load ACLED data from local folder --------
acled = pd.read_csv("/Users/hannahfrank/protest_civil_conflict/data/acled_all_events.csv",low_memory=False,index_col=[0]) 

### Only include certain events -----
df_s = acled.loc[(acled['event_type']=="Protests")].copy(deep=True)

### Add dates ----------
df_s["dd"] = pd.to_datetime(df_s['event_date'],format='%d %B %Y')
df_s["dd"] = df_s["dd"].dt.strftime('%Y-%m')

### Aggregate to month level -------
agg_month = pd.DataFrame(df_s.groupby(["dd","year","iso","country"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"n_protest_events"},inplace=True)

### Get dates and countries --------
# Check coverage here:
# https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2019/01/ACLED_Country-and-Time-Period-coverage_updatedFeb2022.pdf
country_dates={156:[2018,2023], # China, 710
               392:[2018,2023], # Japan, 740
               496:[2018,2023], # Mongolia, 712
               408:[2018,2023], # North Korea, 731
               410:[2018,2023], # South Korea, 732
               158:[2018,2023], # Taiwan, 713
               4:[2017,2023], # Afghanistan, 700
               51:[2018,2023], # Armenia, 371
               31:[2018,2023], # Azerbaijan, 373
               268:[2018,2023], # Georgia, 372
               398:[2018,2023], # Kazakhstan, 705
               417:[2018,2023], # Kyrgyzstan, 703
               762:[2018,2023], # Tajikistan, 702
               795:[2018,2023], # Turkmenistan, 701
               860:[2018,2023], # Uzbekistan, 704
               8:[2018,2023], # Albania, 339
               20:[2020,2023], # Andorra, 232
               40:[2020,2023], # Austria, 305
               831:[2020,2023], # Bailiwick of Guernsey --> not in GW
               832:[2020,2023], # Bailiwick of Jersey --> not in GW
               112:[2018,2023], # Belarus, 370
               56:[2020,2023], # Belgium, 211
               70:[2018,2023], # Bosnia and Herzegovina, 346
               100:[2018,2023], # Bulgaria, 355
               191:[2018,2023], # Croatia, 344
               196:[2018,2023], # Cyprus, 352
               203:[2020,2023], # Czech Republic, 316
               208:[2020,2023], # Denmark, 390
               233:[2020,2023], # Estonia, 366
               234:[2020,2023], # Faroe Islands --> not in GW
               246:[2020,2023], # Finland, 375
               250:[2020,2023], # France, 220
               276:[2020,2023], # Germany, 260
               292:[2020,2023], # Gibraltar --> not in GW
               300:[2018,2023], # Greece, 350
               304:[2020,2023], # Greenland --> not in GW
               348:[2020,2023], # Hungary, 310
               352:[2020,2023], # Iceland, 395
               372:[2020,2023], # Ireland, 205
               833:[2020,2023], # Isle of Man --> not in GW
               380:[2020,2023], # Italy, 325
               0:[2018,2023], # Kosovo, 347
               428:[2020,2023], # Latvia, 367
               438:[2020,2023], # Liechtenstein, 223
               440:[2020,2023], # Lithuania, 368
               442:[2020,2023], # Luxembourg, 212
               470:[2020,2023], # Malta, 338
               498:[2018,2023], # Moldova, 359
               492:[2020,2023], # Monaco, 221
               499:[2018,2023], # Montenegro, 341
               528:[2020,2023], # Netherlands, 210
               807:[2018,2023], # North Macedonia, 343
               578:[2020,2023], # Norway, 385
               616:[2020,2023], # Poland, 290
               620:[2020,2023], # Portugal, 235
               642:[2018,2023], # Romania, 360
               643:[2018,2023], # Russia, 365
               674:[2020,2023], # San Marino, 331
               688:[2018,2023], # Serbia, 340
               703:[2020,2023], # Slovakia, 317
               705:[2020,2023], # Slovenia, 349
               724:[2020,2023], # Spain, 230
               752:[2020,2023], # Sweden, 380
               756:[2020,2023], # Switzerland, 225
               804:[2018,2023], # Ukraine, 369
               826:[2020,2023], # United Kingdom, 200
               336:[2020,2023], # Vatican City --> not in GW
               760:[2017,2023],  # Syria, 652
               368:[2016,2023], # Iraq, 645
               792:[2016,2023], # Turkey, 640
               364:[2016,2023], # Iran, 630
               682:[2015,2023], # Saudi Arabia, 670
               887:[2015,2023], # Yemen, 678
               784:[2016,2023], # United Arab Emirates, 696
               376:[2016,2023], # Israel, 666
               400:[2016,2023], # Jordan, 663
               275:[2016,2023], # Palestine --> not in GW
               422:[2016,2023], # Lebanon, 660
               512:[2016,2023], # Oman, 698
               414:[2016,2023], # Kuwait, 690
               634:[2016,2023], # Qatar, 694
               48:[2016,2023], # Bahrain, 692
               50:[2010,2023], # Bangladesh, 771
               64:[2020,2023], # Bhutan, 760
               96:[2020,2023], # Brunei, 835
               116:[2010,2023], # Cambodia, 811
               626:[2020,2023], # East Timor, 860
               356:[2016,2023], # India, 750
               360:[2015,2023], # Indonesia, 850
               418:[2010,2023], # Laos, 812
               458:[2018,2023], # Malaysia, 820
               462:[2020,2023], # Maldives, 781
               104:[2010,2023], # Myanmar, 775
               524:[2010,2023], # Nepal, 790
               586:[2010,2023], # Pakistan, 770
               608:[2016,2023], # Philippines, 840
               702:[2020,2023], # Singapore, 830
               144:[2010,2023], # Sri Lanka, 780
               764:[2010,2023], # Thailand, 800
               704:[2010,2023], # Vietnam, 816
               12:[1997,2023], # Algeria, 615
               24:[1997,2023],# Angola, 540
               204:[1997,2023], # Benin, 434
               72:[1997,2023], # Botswana, 571
               86:[2018,2023], # British Indian Ocean Territory --> not in GW
               854:[1997,2023], # Burkina Faso, 439
               108:[1997,2023], # Burundi, 516
               120:[1997,2023], # Cameroon, 471
               132:[2020,2023], # Cape Verde, 402
               140:[1997,2023], # Central African Republic, 482
               148:[1997,2023], # Chad, 483
               174:[2020,2023], # Comoros, 581
               180:[1997,2023], # Democratic Republic of Congo, 490
               178:[1997,2023], # Republic of Congo, 484
               384:[1997,2023], # Ivory Coast, 437
               262:[1997,2023], # Djibouti, 522
               818:[1997,2023], # Egypt, 651
               226:[1997,2023], # Equatorial Guinea, 411
               232:[1997,2023], # Eritrea, 531
               748:[1997,2023], # eSwatini, 572
               231:[1997,2023], # Ethiopia, 530
               266:[1997,2023], # Gabon, 481
               270:[1997,2023], # Gambia, 420
               288:[1997,2023], # Ghana, 452
               324:[1997,2023], # Guinea, 438
               624:[1997,2023], # Guinea-Bissau, 404
               404:[1997,2023], # Kenya, 501
               426:[1997,2023], # Lesotho, 570
               430:[1997,2023], # Liberia, 450
               434:[1997,2023], # Libya, 620
               450:[1997,2023], # Madagascar, 580
               454:[1997,2023], # Malawi, 553
               466:[1997,2023], # Mali, 432
               478:[1997,2023], # Mauritania, 435
               480:[2020,2023], # Mauritius, 590
               175:[2020,2023], # Mayotte --> not in GW
               504:[1997,2023], # Morocco, 600
               508:[1997,2023], # Mozambique, 541
               516:[1997,2023], # Namibia, 565
               562:[1997,2023], # Niger, 436
               566:[1997,2023], # Nigeria, 475
               638:[2020,2023], # Réunion --> not in GW
               646:[1997,2023], # Rwanda, 517
               654:[2020,2023], # Saint Helena --> not in GW
               678:[2020,2023], # São Tomé and Príncipe, 403
               686:[1997,2023], # Senegal, 433
               690:[2020,2023], # Seychelles, 591
               694:[1997,2023], # Sierra Leone, 451
               706:[1997,2023], # Somalia, 520
               710:[1997,2023], # South Africa, 560
               728:[2011,2023], # South Sudan, 626
               729:[1997,2023], # Sudan, 625
               834:[1997,2023], # Tanzania, 510
               768:[1997,2023], # Togo, 461
               788:[1997,2023], # Tunisia, 616
               800:[1997,2023], # Uganda, 500
               894:[1997,2023], # Zambia, 551
               716:[1997,2023], # Zimbabwe, 552
               630:[2018,2023], # Puerto Rico (USA) --> not in GW
               312:[2018,2023], # Guadeloupe (France) --> not in GW
               474:[2018,2023], # Martinique (France) --> not in GW
               254:[2018,2023], # French Guiana (France) --> not in GW
               531:[2018,2023], # Curaçao (Netherlands)--> not in GW
               533:[2018,2023], # Aruba (Netherlands) --> not in GW
               850:[2018,2023], # US Virgin Islands (USA) --> not in GW
               136:[2018,2023], # Cayman Islands --> not in GW
               534:[2018,2023], # Sint Maarten (Netherlands) --> not in GW
               796:[2018,2023], # Turks and Caicos (UK) --> not in GW
               92:[2018,2023], # British Virgin Islands (UK) --> not in GW
               535:[2018,2023], # Caribbean Netherlands --> not in GW
               660:[2018,2023], # Anguilla (UK) --> not in GW
               500:[2018,2023], # Montserrat (UK) --> not in GW
               238:[2018,2023], # Falkland Islands (UK) --> not in GW
               76:[2018,2023], # Brazil, 140
               484:[2018,2023], # Mexico, 70
               170:[2018,2023], # Colombia, 100
               32:[2018,2023], # Argentina, 160
               604:[2018,2023], # Peru, 135
               862:[2018,2023], # Venezuela, 101
               152:[2018,2023], # Chile, 155
               320:[2018,2023], # Guatemala, 90
               218:[2018,2023], # Ecuador, 130
               192:[2018,2023], # Cuba, 40
               68:[2018,2023], # Bolivia, 145
               332:[2018,2023], # Haiti, 41
               214:[2018,2023], # Dominican Republic, 42
               340:[2018,2023], # Honduras, 91
               600:[2018,2023], # Paraguay, 150
               222:[2018,2023], # El Salvador, 92
               558:[2018,2023], # Nicaragua, 93
               188:[2018,2023], # Costa Rica, 94
               591:[2018,2023], # Panama, 95
               858:[2018,2023], # Uruguay, 165
               388:[2018,2023], # Jamaica, 51
               780:[2018,2023], # Trinidad and Tobago, 52
               328:[2018,2023], # Guyana, 110
               740:[2018,2023], # Suriname, 115
               44:[2018,2023], # Bahamas, 31
               84:[2018,2023], # Belize, 80
               52:[2018,2023], # Barbados, 53
               662:[2018,2023], # Saint Lucia, 56
               670:[2018,2023], # St. Vincent & Grenadines, 57
               308:[2018,2023], # Grenada, 55
               28:[2018,2023], # Antigua & Barbuda, 58
               212:[2018,2023], # Dominica, 54
               652:[2018,2023], # Saint-Barthelemy --> not in GW
               663:[2018,2023], # Saint-Martin --> not in GW
               659:[2018,2023], # Saint Kitts & Nevis, 60
               239:[2018,2023], # South Georgia and the South Sandwich Islands (UK) --> not in GW
               124:[2021,2023], # Canada, 20
               60:[2021,2023], # Bermuda (UK) --> not in GW
               666:[2021,2023], # Saint Pierre and Miquelon (France) --> not in GW
               840:[2020,2023], # United States, 2
               36:[2021,2023], # Australia, 900
               554:[2021,2023], # New Zealand, 920
               242:[2021,2023], # Fiji, 950
               540:[2021,2023], # New Caledonia (France) --> not in GW
               548:[2021,2023], # Vanuatu, 935
               90:[2021,2023], # Solomon Islands, 940
               598:[2021,2023], # Papua New Guinea, 910
               583:[2021,2023], # Micronesia, 987
               316:[2021,2023], # Guam (US) --> not in GW
               520:[2021,2023], # Nauru, 971
               584:[2021,2023], # Marshall Islands, 983
               296:[2021,2023], # Kiribati, 970
               585:[2021,2023], # Palau, 986
               580:[2021,2023], # Northern Mariana Islands (US) --> not in GW
               16:[2021,2023], # American Samoa (US) --> not in GW
               184:[2021,2023], # Cook Islands --> not in GW
               258:[2021,2023], # French Polynesia (France) --> not in GW
               574:[2021,2023], # Norfolk Island (Australia) --> not in GW
               570:[2021,2023], # Niue --> not in GW
               882:[2021,2023], # Samoa, 990
               776:[2021,2023], # Tonga, 972
               772:[2021,2023], # Tokelau (New Zealand) --> not in GW
               798:[2021,2023], # Tuvalu, 973
               612:[2021,2023], # Pitcairn (UK) --> not in GW
               876:[2021,2023], # Wallis and Futuna (France) --> not in GW
               334:[2021,2023], # Heard Island and McDonald Islands (Australia) --> not in GW
               166:[2021,2023], # Cocos (Keeling) Islands (Australia) --> not in GW
               162:[2021,2023], # Christmas Island (Australia) --> not in GW
               581:[2021,2023], # US Outlying Minor Islands --> not in GW
               10:[2021,2023], # Antarctica --> not in GW, **remove**
               }


### Add missing observation to time series, those have zero fatalities --------
agg_month = agg_month.sort_values(by=["country","year","dd"])

base=pd.DataFrame()
countries=list(country_dates.keys())
# Loop through every country-month
for i in range(0, len(countries)):
    date = list(pd.date_range(start=f"{country_dates[countries[i]][0]}-01",end=f"{country_dates[countries[i]][1]}-12",freq="MS"))
    date = pd.to_datetime(date, format='%Y-%m').to_period('M')
    for x in range(0, len(date)):
            
        # Subset data to add
        s = {'dd':date[x],'iso':countries[i]}
        s = pd.DataFrame(data=s,index=[0])
        base = pd.concat([base,s])  

base = base.sort_values(by=["iso","dd"])
base.reset_index(drop=True,inplace=True)
base["dd"]=base["dd"].astype(str)

agg_month=pd.merge(base, agg_month[["dd","iso","n_protest_events"]],on=["dd","iso"],how="left")
agg_month=agg_month.fillna(0)

# Add country
add_countries = acled[['country', 'iso']].drop_duplicates()
agg_month=pd.merge(agg_month, add_countries,on=["iso"],how="left")

agg_month.loc[agg_month["iso"]==86,"country"]="British Indian Ocean Territory"
agg_month.loc[agg_month["iso"]==166,"country"]="Cocos (Keeling) Islands"
agg_month.loc[agg_month["iso"]==239,"country"]="South Georgia and the South Sandwich Islands"
agg_month.loc[agg_month["iso"]==334,"country"]="Heard Island and McDonald Islands"
agg_month.loc[agg_month["iso"]==574,"country"]="Norfolk Island"
agg_month.loc[agg_month["iso"]==581,"country"]="United States Minor Outlying Islands"
agg_month.loc[agg_month["iso"]==585,"country"]="Palau"
agg_month.loc[agg_month["iso"]==612,"country"]="Pitcairn"
agg_month.loc[agg_month["iso"]==798,"country"]="Tuvalu"

# Add continent
add_countries = acled[['region', 'iso']].drop_duplicates()
agg_month=pd.merge(agg_month, add_countries,on=["iso"],how="left")
agg_month.loc[agg_month["iso"]==86,"region"]="Eastern Africa"
agg_month.loc[agg_month["iso"]==166,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==239,"region"]="South America"
agg_month.loc[agg_month["iso"]==334,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==574,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==581,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==585,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==612,"region"]="Oceania"
agg_month.loc[agg_month["iso"]==798,"region"]="Oceania"

# Remove antacrtica
agg_month=agg_month.loc[agg_month["country"]!="Antarctica"]

### Merge countries which are not independent ###

# "South Georgia and the South Sandwich Islands" -->  United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="South Georgia and the South Sandwich Islands"].max() # Only zeros
agg_month=agg_month.loc[agg_month["country"]!="South Georgia and the South Sandwich Islands"]

# "US Outlying Minor Islands" -->  United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States Minor Outlying Islands"]
agg_month=agg_month.loc[agg_month["country"]!="United States Minor Outlying Islands"] # Only zeros

# "Cocos (Keeling) Islands" -->  Australia
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Cocos (Keeling) Islands"]
agg_month=agg_month.loc[agg_month["country"]!="Cocos (Keeling) Islands"] # Only zeros

# "Heard Island and McDonald Islands" -->  Australia
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Heard Island and McDonald Islands"]
agg_month=agg_month.loc[agg_month["country"]!="Heard Island and McDonald Islands"] # Only zeros

# "Pitcairn" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Pitcairn"]
agg_month=agg_month.loc[agg_month["country"]!="Pitcairn"] # Only zeros

# "Tokelau" --> New Zealand
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Tokelau"]
agg_month=agg_month.loc[agg_month["country"]!="Tokelau"] # Only zeros

# "Niue" --> New Zealand
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Niue"]
agg_month=agg_month.loc[agg_month["country"]!="Niue"] # Only zeros

# "Norfolk Island" --> Australia
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Norfolk Island"]
agg_month=agg_month.loc[agg_month["country"]!="Norfolk Island"] # Only zeros

# "Bermuda" --> United Kingdom 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Bermuda"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom") & (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Bermuda"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom") & (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="Bermuda"]

# "Puerto Rico" --> United States 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Puerto Rico"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Puerto Rico") & (agg_month["dd"]>="2020-01") ].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")].values
agg_month=agg_month.loc[agg_month["country"]!="Puerto Rico"]

# "Falkland Islands" --> France 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Falkland Islands"].max()
agg_month=agg_month.loc[agg_month["country"]!="Falkland Islands"] # Only zeros

# "Reunion" --> France 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Reunion"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Reunion")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Reunion"]

# "British Indian Ocean Territory" --> United Kingdom 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="British Indian Ocean Territory"].max()
agg_month=agg_month.loc[agg_month["country"]!="British Indian Ocean Territory"] # Only zeros

# "Palestine" --> Israel 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Palestine"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Israel"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Israel")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Palestine")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Israel")].values
agg_month=agg_month.loc[agg_month["country"]!="Palestine"]

# "Vatican City" --> Italy 
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Vatican City"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Italy"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Italy")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Vatican City")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Italy")].values
agg_month=agg_month.loc[agg_month["country"]!="Vatican City"]

# "American Samoa"--> United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="American Samoa"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[agg_month["country"]=="American Samoa"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="American Samoa"]

# "Anguilla"--> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Anguilla"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Anguilla")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Anguilla"]

# "Aruba"--> Netherlands
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Aruba"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Netherlands"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Aruba")&(agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")].values
agg_month=agg_month.loc[agg_month["country"]!="Aruba"]

# "Cook Islands" --> New Zealand
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Cook Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="New Zealand"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="New Zealand")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Cook Islands")&(agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="New Zealand")].values
agg_month=agg_month.loc[agg_month["country"]!="Cook Islands"]

# "Bailiwick of Guernsey" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Bailiwick of Guernsey"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Bailiwick of Guernsey"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Bailiwick of Guernsey"]

# "Bailiwick of Jersey" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Bailiwick of Jersey"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Bailiwick of Jersey"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Bailiwick of Jersey"]

# "British Virgin Islands" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="British Virgin Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="British Virgin Islands")&(agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="British Virgin Islands"]

# "Caribbean Netherlands" --> Netherlands
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Caribbean Netherlands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Netherlands"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Caribbean Netherlands")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")].values
agg_month=agg_month.loc[agg_month["country"]!="Caribbean Netherlands"]

# "Cayman Islands" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Cayman Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Cayman Islands")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Cayman Islands"]

# "Christmas Island" --> Australia
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Christmas Island"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Australia"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Australia")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Christmas Island")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Australia")].values
agg_month=agg_month.loc[agg_month["country"]!="Christmas Island"]

# "Curacao" --> Netherlands
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Curacao"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Netherlands"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Curacao")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")].values
agg_month=agg_month.loc[agg_month["country"]!="Curacao"]

# "Faroe Islands" --> Denmark
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Faroe Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Denmark"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Denmark")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Faroe Islands")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Denmark")].values
agg_month=agg_month.loc[agg_month["country"]!="Faroe Islands"]

# "French Guiana" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="French Guiana"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="French Guiana")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="French Guiana"]

# "French Polynesia" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="French Polynesia"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="French Polynesia")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="French Polynesia"]

# "Gibraltar" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Gibraltar"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Gibraltar")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Gibraltar"]

# "Greenland" --> Denmark
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Greenland"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Denmark"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Denmark")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Greenland")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Denmark")].values
agg_month=agg_month.loc[agg_month["country"]!="Greenland"]

# "Guadeloupe" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Guadeloupe"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Guadeloupe")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Guadeloupe"]

# "Guam" --> United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Guam"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Guam"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="Guam"]

# "Isle of Man" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Isle of Man"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Isle of Man")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Isle of Man"]

# "Martinique" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Martinique"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Martinique")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Martinique"]

# "Mayotte" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Mayotte"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Mayotte")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Mayotte"]

# "Montserrat" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Montserrat"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Montserrat")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Montserrat"]

# "New Caledonia" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="New Caledonia"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="New Caledonia")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="New Caledonia"]

# "Northern Mariana Islands" --> United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Northern Mariana Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Northern Mariana Islands"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="Northern Mariana Islands"]

# "Saint-Barthelemy" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Saint-Barthelemy"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[agg_month["country"]=="France"]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Saint-Barthelemy")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Saint-Barthelemy"]

# "Saint Helena" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Saint Helena, Ascension and Tristan da Cunha"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United Kingdom"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Saint Helena, Ascension and Tristan da Cunha")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United Kingdom")].values
agg_month=agg_month.loc[agg_month["country"]!="Saint Helena, Ascension and Tristan da Cunha"]

# "Saint Pierre and Miquelon" --> United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Saint Pierre and Miquelon"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[agg_month["country"]=="Saint Pierre and Miquelon"].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States") & (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="Saint Pierre and Miquelon"]

# "Saint-Martin" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Saint-Martin"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Saint-Martin")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")].values
agg_month=agg_month.loc[agg_month["country"]!="Saint-Martin"]

# "Sint Maarten" --> Netherlands
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Sint Maarten"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Netherlands"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Sint Maarten")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="Netherlands")].values
agg_month=agg_month.loc[agg_month["country"]!="Sint Maarten"]

# "Turks and Caicos Islands" --> United Kingdom
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Turks and Caicos Islands"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Turks and Caicos Islands")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")].values
agg_month=agg_month.loc[agg_month["country"]!="Turks and Caicos Islands"]

# "Virgin Islands, U.S." --> United States
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Virgin Islands, U.S."]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="United States"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Virgin Islands, U.S.")& (agg_month["dd"]>="2020-01")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="United States")].values
agg_month=agg_month.loc[agg_month["country"]!="Virgin Islands, U.S."]

# "Wallis and Futuna" --> France
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="Wallis and Futuna"]
agg_month[["dd","n_protest_events"]].loc[agg_month["country"]=="France"]
agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")]=agg_month["n_protest_events"].loc[(agg_month["country"]=="Wallis and Futuna")].values+agg_month["n_protest_events"].loc[(agg_month["country"]=="France")& (agg_month["dd"]>="2021-01")].values
agg_month=agg_month.loc[agg_month["country"]!="Wallis and Futuna"]


agg_month = agg_month.sort_values(by=["country","dd"])


### Country codes #####
agg_month.country.unique()

agg_month["gw_codes"]=999999
agg_month.loc[agg_month["country"]=="Afghanistan","gw_codes"]=700
agg_month.loc[agg_month["country"]=="Albania","gw_codes"]=339
agg_month.loc[agg_month["country"]=="Algeria","gw_codes"]=615
agg_month.loc[agg_month["country"]=="Andorra","gw_codes"]=232
agg_month.loc[agg_month["country"]=="Angola","gw_codes"]=540
agg_month.loc[agg_month["country"]=="Antigua and Barbuda","gw_codes"]=58
agg_month.loc[agg_month["country"]=="Argentina","gw_codes"]=160
agg_month.loc[agg_month["country"]=="Armenia","gw_codes"]=371
agg_month.loc[agg_month["country"]=="Australia","gw_codes"]=900
agg_month.loc[agg_month["country"]=="Austria","gw_codes"]=305
agg_month.loc[agg_month["country"]=="Azerbaijan","gw_codes"]=373

agg_month.loc[agg_month["country"]=="Bahamas","gw_codes"]=31
agg_month.loc[agg_month["country"]=="Bahrain","gw_codes"]=692
agg_month.loc[agg_month["country"]=="Bangladesh","gw_codes"]=771
agg_month.loc[agg_month["country"]=="Barbados","gw_codes"]=53
agg_month.loc[agg_month["country"]=="Belarus","gw_codes"]=370
agg_month.loc[agg_month["country"]=="Belgium","gw_codes"]=211
agg_month.loc[agg_month["country"]=="Belize","gw_codes"]=80
agg_month.loc[agg_month["country"]=="Benin","gw_codes"]=434
agg_month.loc[agg_month["country"]=="Bhutan","gw_codes"]=760
agg_month.loc[agg_month["country"]=="Bolivia","gw_codes"]=145
agg_month.loc[agg_month["country"]=="Bosnia and Herzegovina","gw_codes"]=346
agg_month.loc[agg_month["country"]=="Botswana","gw_codes"]=571
agg_month.loc[agg_month["country"]=="Brazil","gw_codes"]=140
agg_month.loc[agg_month["country"]=="Brunei","gw_codes"]=835
agg_month.loc[agg_month["country"]=="Bulgaria","gw_codes"]=355
agg_month.loc[agg_month["country"]=="Burkina Faso","gw_codes"]=439
agg_month.loc[agg_month["country"]=="Burundi","gw_codes"]=516

agg_month.loc[agg_month["country"]=="Cape Verde","gw_codes"]=402
agg_month.loc[agg_month["country"]=="Cambodia","gw_codes"]=811
agg_month.loc[agg_month["country"]=="Cameroon","gw_codes"]=471
agg_month.loc[agg_month["country"]=="Canada","gw_codes"]=20
agg_month.loc[agg_month["country"]=="Central African Republic","gw_codes"]=482
agg_month.loc[agg_month["country"]=="Chad","gw_codes"]=483
agg_month.loc[agg_month["country"]=="Chile","gw_codes"]=155
agg_month.loc[agg_month["country"]=="China","gw_codes"]=710
agg_month.loc[agg_month["country"]=="Colombia","gw_codes"]=100
agg_month.loc[agg_month["country"]=="Comoros","gw_codes"]=581
agg_month.loc[agg_month["country"]=="Costa Rica","gw_codes"]=94
agg_month.loc[agg_month["country"]=="Croatia","gw_codes"]=344
agg_month.loc[agg_month["country"]=="Cuba","gw_codes"]=40
agg_month.loc[agg_month["country"]=="Cyprus","gw_codes"]=352
agg_month.loc[agg_month["country"]=="Czech Republic","gw_codes"]=316

agg_month.loc[agg_month["country"]=="Democratic Republic of Congo","gw_codes"]=490
agg_month.loc[agg_month["country"]=="Denmark","gw_codes"]=390
agg_month.loc[agg_month["country"]=="Djibouti","gw_codes"]=522
agg_month.loc[agg_month["country"]=="Dominica","gw_codes"]=54
agg_month.loc[agg_month["country"]=="Dominican Republic","gw_codes"]=42

agg_month.loc[agg_month["country"]=="East Timor","gw_codes"]=860
agg_month.loc[agg_month["country"]=="Ecuador","gw_codes"]=130
agg_month.loc[agg_month["country"]=="Egypt","gw_codes"]=651
agg_month.loc[agg_month["country"]=="El Salvador","gw_codes"]=92
agg_month.loc[agg_month["country"]=="Equatorial Guinea","gw_codes"]=411
agg_month.loc[agg_month["country"]=="Eritrea","gw_codes"]=531
agg_month.loc[agg_month["country"]=="Estonia","gw_codes"]=366
agg_month.loc[agg_month["country"]=="Ethiopia","gw_codes"]=530

agg_month.loc[agg_month["country"]=="Fiji","gw_codes"]=950
agg_month.loc[agg_month["country"]=="Finland","gw_codes"]=375
agg_month.loc[agg_month["country"]=="France","gw_codes"]=220

agg_month.loc[agg_month["country"]=="Gabon","gw_codes"]=481
agg_month.loc[agg_month["country"]=="Gambia","gw_codes"]=420
agg_month.loc[agg_month["country"]=="Georgia","gw_codes"]=372
agg_month.loc[agg_month["country"]=="Germany","gw_codes"]=260
agg_month.loc[agg_month["country"]=="Ghana","gw_codes"]=452
agg_month.loc[agg_month["country"]=="Greece","gw_codes"]=350
agg_month.loc[agg_month["country"]=="Grenada","gw_codes"]=55
agg_month.loc[agg_month["country"]=="Guatemala","gw_codes"]=90
agg_month.loc[agg_month["country"]=="Guinea","gw_codes"]=438
agg_month.loc[agg_month["country"]=="Guinea-Bissau","gw_codes"]=404
agg_month.loc[agg_month["country"]=="Guyana","gw_codes"]=110

agg_month.loc[agg_month["country"]=="Haiti","gw_codes"]=41
agg_month.loc[agg_month["country"]=="Honduras","gw_codes"]=91
agg_month.loc[agg_month["country"]=="Hungary","gw_codes"]=310

agg_month.loc[agg_month["country"]=="Iceland","gw_codes"]=395
agg_month.loc[agg_month["country"]=="India","gw_codes"]=750
agg_month.loc[agg_month["country"]=="Indonesia","gw_codes"]=850
agg_month.loc[agg_month["country"]=="Iran","gw_codes"]=630
agg_month.loc[agg_month["country"]=="Iraq","gw_codes"]=645
agg_month.loc[agg_month["country"]=="Ireland","gw_codes"]=205
agg_month.loc[agg_month["country"]=="Israel","gw_codes"]=666
agg_month.loc[agg_month["country"]=="Italy","gw_codes"]=325
agg_month.loc[agg_month["country"]=="Ivory Coast","gw_codes"]=437

agg_month.loc[agg_month["country"]=="Jamaica","gw_codes"]=51
agg_month.loc[agg_month["country"]=="Japan","gw_codes"]=740
agg_month.loc[agg_month["country"]=="Jordan","gw_codes"]=663

agg_month.loc[agg_month["country"]=="Kazakhstan","gw_codes"]=705
agg_month.loc[agg_month["country"]=="Kenya","gw_codes"]=501
agg_month.loc[agg_month["country"]=="Kiribati","gw_codes"]=970
agg_month.loc[agg_month["country"]=="Kosovo","gw_codes"]=347
agg_month.loc[agg_month["country"]=="Kuwait","gw_codes"]=690
agg_month.loc[agg_month["country"]=="Kyrgyzstan","gw_codes"]=703

agg_month.loc[agg_month["country"]=="Laos","gw_codes"]=812
agg_month.loc[agg_month["country"]=="Latvia","gw_codes"]=367
agg_month.loc[agg_month["country"]=="Lebanon","gw_codes"]=660
agg_month.loc[agg_month["country"]=="Lesotho","gw_codes"]=570
agg_month.loc[agg_month["country"]=="Liberia","gw_codes"]=450
agg_month.loc[agg_month["country"]=="Libya","gw_codes"]=620
agg_month.loc[agg_month["country"]=="Liechtenstein","gw_codes"]=223
agg_month.loc[agg_month["country"]=="Lithuania","gw_codes"]=368
agg_month.loc[agg_month["country"]=="Luxembourg","gw_codes"]=212

agg_month.loc[agg_month["country"]=="Madagascar","gw_codes"]=580
agg_month.loc[agg_month["country"]=="Malawi","gw_codes"]=553
agg_month.loc[agg_month["country"]=="Malaysia","gw_codes"]=820
agg_month.loc[agg_month["country"]=="Maldives","gw_codes"]=781
agg_month.loc[agg_month["country"]=="Mali","gw_codes"]=432
agg_month.loc[agg_month["country"]=="Malta","gw_codes"]=338
agg_month.loc[agg_month["country"]=="Marshall Islands","gw_codes"]=983
agg_month.loc[agg_month["country"]=="Mauritius","gw_codes"]=590
agg_month.loc[agg_month["country"]=="Mauritania","gw_codes"]=435
agg_month.loc[agg_month["country"]=="Mexico","gw_codes"]=70
agg_month.loc[agg_month["country"]=="Micronesia","gw_codes"]=987
agg_month.loc[agg_month["country"]=="Moldova","gw_codes"]=359
agg_month.loc[agg_month["country"]=="Monaco","gw_codes"]=221
agg_month.loc[agg_month["country"]=="Mongolia","gw_codes"]=712
agg_month.loc[agg_month["country"]=="Montenegro","gw_codes"]=341
agg_month.loc[agg_month["country"]=="Morocco","gw_codes"]=600
agg_month.loc[agg_month["country"]=="Mozambique","gw_codes"]=541
agg_month.loc[agg_month["country"]=="Myanmar","gw_codes"]=775


agg_month.loc[agg_month["country"]=="Namibia","gw_codes"]=565
agg_month.loc[agg_month["country"]=="Nauru","gw_codes"]=971
agg_month.loc[agg_month["country"]=="Nepal","gw_codes"]=790
agg_month.loc[agg_month["country"]=="Netherlands","gw_codes"]=210
agg_month.loc[agg_month["country"]=="New Zealand","gw_codes"]=920
agg_month.loc[agg_month["country"]=="Nicaragua","gw_codes"]=93
agg_month.loc[agg_month["country"]=="Niger","gw_codes"]=436
agg_month.loc[agg_month["country"]=="Nigeria","gw_codes"]=475
agg_month.loc[agg_month["country"]=="North Korea","gw_codes"]=731
agg_month.loc[agg_month["country"]=="North Macedonia","gw_codes"]=343
agg_month.loc[agg_month["country"]=="Norway","gw_codes"]=385

agg_month.loc[agg_month["country"]=="Oman","gw_codes"]=698

agg_month.loc[agg_month["country"]=="Palau","gw_codes"]=986
agg_month.loc[agg_month["country"]=="Pakistan","gw_codes"]=770
agg_month.loc[agg_month["country"]=="Panama","gw_codes"]=95
agg_month.loc[agg_month["country"]=="Papua New Guinea","gw_codes"]=910
agg_month.loc[agg_month["country"]=="Paraguay","gw_codes"]=150
agg_month.loc[agg_month["country"]=="Peru","gw_codes"]=135
agg_month.loc[agg_month["country"]=="Philippines","gw_codes"]=840
agg_month.loc[agg_month["country"]=="Poland","gw_codes"]=290
agg_month.loc[agg_month["country"]=="Portugal","gw_codes"]=235

agg_month.loc[agg_month["country"]=="Qatar","gw_codes"]=694

agg_month.loc[agg_month["country"]=="Romania","gw_codes"]=360
agg_month.loc[agg_month["country"]=="Russia","gw_codes"]=365
agg_month.loc[agg_month["country"]=="Rwanda","gw_codes"]=517
agg_month.loc[agg_month["country"]=="Republic of Congo","gw_codes"]=484

agg_month.loc[agg_month["country"]=="Saint Kitts and Nevis","gw_codes"]=60
agg_month.loc[agg_month["country"]=="Saint Lucia","gw_codes"]=56
agg_month.loc[agg_month["country"]=="Saint Vincent and the Grenadines","gw_codes"]=57
agg_month.loc[agg_month["country"]=="Samoa","gw_codes"]=990
agg_month.loc[agg_month["country"]=="San Marino","gw_codes"]=331
agg_month.loc[agg_month["country"]=="Sao Tome and Principe","gw_codes"]=403
agg_month.loc[agg_month["country"]=="Saudi Arabia","gw_codes"]=670
agg_month.loc[agg_month["country"]=="Senegal","gw_codes"]=433
agg_month.loc[agg_month["country"]=="Serbia","gw_codes"]=340
agg_month.loc[agg_month["country"]=="Seychelles","gw_codes"]=591
agg_month.loc[agg_month["country"]=="Sierra Leone","gw_codes"]=451
agg_month.loc[agg_month["country"]=="Singapore","gw_codes"]=830
agg_month.loc[agg_month["country"]=="Slovakia","gw_codes"]=317
agg_month.loc[agg_month["country"]=="Slovenia","gw_codes"]=349
agg_month.loc[agg_month["country"]=="Solomon Islands","gw_codes"]=940
agg_month.loc[agg_month["country"]=="Somalia","gw_codes"]=520
agg_month.loc[agg_month["country"]=="South Africa","gw_codes"]=560
agg_month.loc[agg_month["country"]=="South Korea","gw_codes"]=732
agg_month.loc[agg_month["country"]=="South Sudan","gw_codes"]=626
agg_month.loc[agg_month["country"]=="Spain","gw_codes"]=230
agg_month.loc[agg_month["country"]=="Sri Lanka","gw_codes"]=780
agg_month.loc[agg_month["country"]=="Sudan","gw_codes"]=625
agg_month.loc[agg_month["country"]=="Suriname","gw_codes"]=115
agg_month.loc[agg_month["country"]=="Sweden","gw_codes"]=380
agg_month.loc[agg_month["country"]=="Switzerland","gw_codes"]=225
agg_month.loc[agg_month["country"]=="Syria","gw_codes"]=652

agg_month.loc[agg_month["country"]=="Taiwan","gw_codes"]=713
agg_month.loc[agg_month["country"]=="Tajikistan","gw_codes"]=702
agg_month.loc[agg_month["country"]=="Tanzania","gw_codes"]=510
agg_month.loc[agg_month["country"]=="Thailand","gw_codes"]=800
agg_month.loc[agg_month["country"]=="Togo","gw_codes"]=461
agg_month.loc[agg_month["country"]=="Tonga","gw_codes"]=972
agg_month.loc[agg_month["country"]=="Trinidad and Tobago","gw_codes"]=52
agg_month.loc[agg_month["country"]=="Tunisia","gw_codes"]=616
agg_month.loc[agg_month["country"]=="Turkey","gw_codes"]=640
agg_month.loc[agg_month["country"]=="Turkmenistan","gw_codes"]=701
agg_month.loc[agg_month["country"]=="Tuvalu","gw_codes"]=973

agg_month.loc[agg_month["country"]=="Uganda","gw_codes"]=500
agg_month.loc[agg_month["country"]=="Ukraine","gw_codes"]=369
agg_month.loc[agg_month["country"]=="United Arab Emirates","gw_codes"]=696
agg_month.loc[agg_month["country"]=="United Kingdom","gw_codes"]=200
agg_month.loc[agg_month["country"]=="United States","gw_codes"]=2
agg_month.loc[agg_month["country"]=="Uruguay","gw_codes"]=165
agg_month.loc[agg_month["country"]=="Uzbekistan","gw_codes"]=704

agg_month.loc[agg_month["country"]=="Vanuatu","gw_codes"]=935
agg_month.loc[agg_month["country"]=="Venezuela","gw_codes"]=101
agg_month.loc[agg_month["country"]=="Vietnam","gw_codes"]=816

agg_month.loc[agg_month["country"]=="Yemen","gw_codes"]=678
agg_month.loc[agg_month["country"]=="Zambia","gw_codes"]=551
agg_month.loc[agg_month["country"]=="Zimbabwe","gw_codes"]=552

agg_month.loc[agg_month["country"]=="eSwatini","gw_codes"]=572

agg_month['year'] = agg_month['dd'].str[:4].astype(int)

df=agg_month[["dd","year","gw_codes","country","n_protest_events","region"]]

                                ############
                                ### UCDP ###
                                ############
                                
ucdp = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged231-csv.zip",low_memory=False)
ucdp2 = pd.read_csv("https://ucdp.uu.se/downloads/candidateged/GEDEvent_v23_01_23_12.csv",low_memory=False)
ucdp=pd.concat([ucdp,ucdp2],axis=0,ignore_index=True)

### Only use state-based violence ---------
ucdp_s = ucdp[(ucdp["type_of_violence"]==1)].copy(deep=True)

### Exclude state-based between governments ----------
ucdp_ss = ucdp_s.loc[(ucdp_s["dyad_name"] != "Government of Afghanistan - Government of United Kingdom, Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Cambodia (Kampuchea) - Government of Thailand") &
                    (ucdp_s["dyad_name"] != "Government of Cameroon - Government of Nigeria") &
                    (ucdp_s["dyad_name"] != "Government of Djibouti - Government of Eritrea") &
                    (ucdp_s["dyad_name"] != "Government of Ecuador - Government of Peru") &
                    (ucdp_s["dyad_name"] != "Government of Eritrea - Government of Ethiopia") &
                    (ucdp_s["dyad_name"] != "Government of India - Government of Pakistan") &
                    (ucdp_s["dyad_name"] != "Government of China - Government of India") &  
                    (ucdp_s["dyad_name"] != "Government of Iran - Government of Israel") &                    
                    (ucdp_s["dyad_name"] != "Government of Iraq - Government of Kuwait") &
                    (ucdp_s["dyad_name"] != "Government of Australia, Government of United Kingdom, Government of United States of America - Government of Iraq") &
                    (ucdp_s["dyad_name"] != "Government of Kyrgyzstan - Government of Tajikistan") &                    
                    (ucdp_s["dyad_name"] != "Government of Panama - Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Russia (Soviet Union) - Government of Ukraine") &                   
                    (ucdp_s["dyad_name"] != "Government of South Sudan - Government of Sudan") ].copy(deep=True)

### Add dates ----
ucdp_ss["dd_date_start"] = pd.to_datetime(ucdp_ss['date_start'],format='%Y-%m-%d %H:%M:%S.000')
ucdp_ss["dd_date_end"] = pd.to_datetime(ucdp_ss['date_end'],format='%Y-%m-%d %H:%M:%S.000')

# Only store month
ucdp_ss["month_date_start"] = ucdp_ss["dd_date_start"].dt.strftime('%m')
ucdp_ss["month_date_end"] = ucdp_ss["dd_date_end"].dt.strftime('%m')
ucdp_date = ucdp_ss[["year","dd_date_start","dd_date_end","active_year","country","country_id","date_prec","best","deaths_a","deaths_b","deaths_civilians","deaths_unknown","month_date_start","month_date_end"]].copy(deep=True)

# Reset index 
ucdp_date = ucdp_date.sort_values(by=["country", "year"],ascending=True)
ucdp_date.reset_index(drop=True, inplace=True)

### Loop through data and delete observations which comprise more than one month ------
ucdp_final = ucdp_date.copy()
for i in range(0,len(ucdp_date)):
    if ucdp_date["month_date_start"].loc[i]!=ucdp_date["month_date_end"].loc[i]:
        ucdp_final = ucdp_final.drop(index=i, axis=0)        

### Generate year_month variable --------
ucdp_final['dd'] = pd.to_datetime(ucdp_final['dd_date_start'],format='%Y-%m').dt.to_period('M')

### Aggregate fatality variables ----------
fat = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['best'].sum())
ucdp_fat = fat.reset_index()
ucdp_fat.columns=["dd","gw_codes","fatalities"]
ucdp_fat["dd"]=ucdp_fat["dd"].astype(str)

# Merge 
df=pd.merge(df,ucdp_fat,on=["dd","gw_codes"],how="left")
df['fatalities'] = df['fatalities'].fillna(0)

                                    #############
                                    ### V-dem ###
                                    #############
                                    
vdem = pd.read_csv("/Users/hannahfrank/protest_civil_conflict/data/V-Dem-CY-Full+Others-v14.csv",low_memory=False)
        
vdem_s=vdem[["year","country_name","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_neopat","v2x_civlib","v2x_clphy","v2x_corr","v2x_rule"]]   
vdem_s.columns=["year","country","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_neopat","v2x_civlib","v2x_clphy","v2x_corr","v2x_rule"]                           
                         
vdem_s=vdem_s.sort_values(by=["country","year"])

vdem_s["gw_codes"]=999999
vdem_s.loc[vdem_s["country"]=="Afghanistan","gw_codes"]=700
vdem_s.loc[vdem_s["country"]=="Albania","gw_codes"]=339
vdem_s.loc[vdem_s["country"]=="Algeria","gw_codes"]=615
vdem_s.loc[vdem_s["country"]=="Andorra","gw_codes"]=232
vdem_s.loc[vdem_s["country"]=="Angola","gw_codes"]=540
vdem_s.loc[vdem_s["country"]=="Antigua and Barbuda","gw_codes"]=58
vdem_s.loc[vdem_s["country"]=="Argentina","gw_codes"]=160
vdem_s.loc[vdem_s["country"]=="Armenia","gw_codes"]=371
vdem_s.loc[vdem_s["country"]=="Australia","gw_codes"]=900
vdem_s.loc[vdem_s["country"]=="Austria","gw_codes"]=305
vdem_s.loc[vdem_s["country"]=="Azerbaijan","gw_codes"]=373

vdem_s.loc[vdem_s["country"]=="Bahrain","gw_codes"]=692
vdem_s.loc[vdem_s["country"]=="Bangladesh","gw_codes"]=771
vdem_s.loc[vdem_s["country"]=="Barbados","gw_codes"]=53
vdem_s.loc[vdem_s["country"]=="Belarus","gw_codes"]=370
vdem_s.loc[vdem_s["country"]=="Belgium","gw_codes"]=211
vdem_s.loc[vdem_s["country"]=="Belize","gw_codes"]=80
vdem_s.loc[vdem_s["country"]=="Benin","gw_codes"]=434
vdem_s.loc[vdem_s["country"]=="Bhutan","gw_codes"]=760
vdem_s.loc[vdem_s["country"]=="Bolivia","gw_codes"]=145
vdem_s.loc[vdem_s["country"]=="Bosnia and Herzegovina","gw_codes"]=346
vdem_s.loc[vdem_s["country"]=="Botswana","gw_codes"]=571
vdem_s.loc[vdem_s["country"]=="Brazil","gw_codes"]=140
vdem_s.loc[vdem_s["country"]=="Brunei","gw_codes"]=835
vdem_s.loc[vdem_s["country"]=="Bulgaria","gw_codes"]=355
vdem_s.loc[vdem_s["country"]=="Burkina Faso","gw_codes"]=439
vdem_s.loc[vdem_s["country"]=="Burundi","gw_codes"]=516

vdem_s.loc[vdem_s["country"]=="Cape Verde","gw_codes"]=402
vdem_s.loc[vdem_s["country"]=="Cambodia","gw_codes"]=811
vdem_s.loc[vdem_s["country"]=="Cameroon","gw_codes"]=471
vdem_s.loc[vdem_s["country"]=="Canada","gw_codes"]=20
vdem_s.loc[vdem_s["country"]=="Central African Republic","gw_codes"]=482
vdem_s.loc[vdem_s["country"]=="Chad","gw_codes"]=483
vdem_s.loc[vdem_s["country"]=="Chile","gw_codes"]=155
vdem_s.loc[vdem_s["country"]=="China","gw_codes"]=710
vdem_s.loc[vdem_s["country"]=="Colombia","gw_codes"]=100
vdem_s.loc[vdem_s["country"]=="Comoros","gw_codes"]=581
vdem_s.loc[vdem_s["country"]=="Costa Rica","gw_codes"]=94
vdem_s.loc[vdem_s["country"]=="Croatia","gw_codes"]=344
vdem_s.loc[vdem_s["country"]=="Cuba","gw_codes"]=40
vdem_s.loc[vdem_s["country"]=="Cyprus","gw_codes"]=352
vdem_s.loc[vdem_s["country"]=="Czechia","gw_codes"]=316

vdem_s.loc[vdem_s["country"]=="Democratic Republic of the Congo","gw_codes"]=490
vdem_s.loc[vdem_s["country"]=="Denmark","gw_codes"]=390
vdem_s.loc[vdem_s["country"]=="Djibouti","gw_codes"]=522
vdem_s.loc[vdem_s["country"]=="Dominica","gw_codes"]=54
vdem_s.loc[vdem_s["country"]=="Dominican Republic","gw_codes"]=42

vdem_s.loc[vdem_s["country"]=="Timor-Leste","gw_codes"]=860
vdem_s.loc[vdem_s["country"]=="Ecuador","gw_codes"]=130
vdem_s.loc[vdem_s["country"]=="Egypt","gw_codes"]=651
vdem_s.loc[vdem_s["country"]=="El Salvador","gw_codes"]=92
vdem_s.loc[vdem_s["country"]=="Equatorial Guinea","gw_codes"]=411
vdem_s.loc[vdem_s["country"]=="Eritrea","gw_codes"]=531
vdem_s.loc[vdem_s["country"]=="Estonia","gw_codes"]=366
vdem_s.loc[vdem_s["country"]=="Ethiopia","gw_codes"]=530

vdem_s.loc[vdem_s["country"]=="Fiji","gw_codes"]=950
vdem_s.loc[vdem_s["country"]=="Finland","gw_codes"]=375
vdem_s.loc[vdem_s["country"]=="France","gw_codes"]=220

vdem_s.loc[vdem_s["country"]=="Gabon","gw_codes"]=481
vdem_s.loc[vdem_s["country"]=="The Gambia","gw_codes"]=420
vdem_s.loc[vdem_s["country"]=="Georgia","gw_codes"]=372
vdem_s.loc[vdem_s["country"]=="Germany","gw_codes"]=260
vdem_s.loc[vdem_s["country"]=="Ghana","gw_codes"]=452
vdem_s.loc[vdem_s["country"]=="Greece","gw_codes"]=350
vdem_s.loc[vdem_s["country"]=="Grenada","gw_codes"]=55
vdem_s.loc[vdem_s["country"]=="Guatemala","gw_codes"]=90
vdem_s.loc[vdem_s["country"]=="Guinea","gw_codes"]=438
vdem_s.loc[vdem_s["country"]=="Guinea-Bissau","gw_codes"]=404
vdem_s.loc[vdem_s["country"]=="Guyana","gw_codes"]=110

vdem_s.loc[vdem_s["country"]=="Haiti","gw_codes"]=41
vdem_s.loc[vdem_s["country"]=="Honduras","gw_codes"]=91
vdem_s.loc[vdem_s["country"]=="Hungary","gw_codes"]=310

vdem_s.loc[vdem_s["country"]=="Iceland","gw_codes"]=395
vdem_s.loc[vdem_s["country"]=="India","gw_codes"]=750
vdem_s.loc[vdem_s["country"]=="Indonesia","gw_codes"]=850
vdem_s.loc[vdem_s["country"]=="Iran","gw_codes"]=630
vdem_s.loc[vdem_s["country"]=="Iraq","gw_codes"]=645
vdem_s.loc[vdem_s["country"]=="Ireland","gw_codes"]=205
vdem_s.loc[vdem_s["country"]=="Israel","gw_codes"]=666
vdem_s.loc[vdem_s["country"]=="Italy","gw_codes"]=325
vdem_s.loc[vdem_s["country"]=="Ivory Coast","gw_codes"]=437

vdem_s.loc[vdem_s["country"]=="Jamaica","gw_codes"]=51
vdem_s.loc[vdem_s["country"]=="Japan","gw_codes"]=740
vdem_s.loc[vdem_s["country"]=="Jordan","gw_codes"]=663

vdem_s.loc[vdem_s["country"]=="Kazakhstan","gw_codes"]=705
vdem_s.loc[vdem_s["country"]=="Kenya","gw_codes"]=501
vdem_s.loc[vdem_s["country"]=="Kosovo","gw_codes"]=347
vdem_s.loc[vdem_s["country"]=="Kuwait","gw_codes"]=690
vdem_s.loc[vdem_s["country"]=="Kyrgyzstan","gw_codes"]=703

vdem_s.loc[vdem_s["country"]=="Laos","gw_codes"]=812
vdem_s.loc[vdem_s["country"]=="Latvia","gw_codes"]=367
vdem_s.loc[vdem_s["country"]=="Lebanon","gw_codes"]=660
vdem_s.loc[vdem_s["country"]=="Lesotho","gw_codes"]=570
vdem_s.loc[vdem_s["country"]=="Liberia","gw_codes"]=450
vdem_s.loc[vdem_s["country"]=="Libya","gw_codes"]=620
vdem_s.loc[vdem_s["country"]=="Lithuania","gw_codes"]=368
vdem_s.loc[vdem_s["country"]=="Luxembourg","gw_codes"]=212

vdem_s.loc[vdem_s["country"]=="Madagascar","gw_codes"]=580
vdem_s.loc[vdem_s["country"]=="Malawi","gw_codes"]=553
vdem_s.loc[vdem_s["country"]=="Malaysia","gw_codes"]=820
vdem_s.loc[vdem_s["country"]=="Maldives","gw_codes"]=781
vdem_s.loc[vdem_s["country"]=="Mali","gw_codes"]=432
vdem_s.loc[vdem_s["country"]=="Malta","gw_codes"]=338
vdem_s.loc[vdem_s["country"]=="Mauritius","gw_codes"]=590
vdem_s.loc[vdem_s["country"]=="Mauritania","gw_codes"]=435
vdem_s.loc[vdem_s["country"]=="Mexico","gw_codes"]=70
vdem_s.loc[vdem_s["country"]=="Moldova","gw_codes"]=359
vdem_s.loc[vdem_s["country"]=="Mongolia","gw_codes"]=712
vdem_s.loc[vdem_s["country"]=="Montenegro","gw_codes"]=341
vdem_s.loc[vdem_s["country"]=="Morocco","gw_codes"]=600
vdem_s.loc[vdem_s["country"]=="Mozambique","gw_codes"]=541
vdem_s.loc[vdem_s["country"]=="Burma/Myanmar","gw_codes"]=775

vdem_s.loc[vdem_s["country"]=="Namibia","gw_codes"]=565
vdem_s.loc[vdem_s["country"]=="Nepal","gw_codes"]=790
vdem_s.loc[vdem_s["country"]=="Netherlands","gw_codes"]=210
vdem_s.loc[vdem_s["country"]=="New Zealand","gw_codes"]=920
vdem_s.loc[vdem_s["country"]=="Nicaragua","gw_codes"]=93
vdem_s.loc[vdem_s["country"]=="Niger","gw_codes"]=436
vdem_s.loc[vdem_s["country"]=="Nigeria","gw_codes"]=475
vdem_s.loc[vdem_s["country"]=="North Korea","gw_codes"]=731
vdem_s.loc[vdem_s["country"]=="North Macedonia","gw_codes"]=343
vdem_s.loc[vdem_s["country"]=="Norway","gw_codes"]=385

vdem_s.loc[vdem_s["country"]=="Oman","gw_codes"]=698

vdem_s.loc[vdem_s["country"]=="Pakistan","gw_codes"]=770
vdem_s.loc[vdem_s["country"]=="Panama","gw_codes"]=95
vdem_s.loc[vdem_s["country"]=="Papua New Guinea","gw_codes"]=910
vdem_s.loc[vdem_s["country"]=="Paraguay","gw_codes"]=150
vdem_s.loc[vdem_s["country"]=="Peru","gw_codes"]=135
vdem_s.loc[vdem_s["country"]=="Philippines","gw_codes"]=840
vdem_s.loc[vdem_s["country"]=="Poland","gw_codes"]=290
vdem_s.loc[vdem_s["country"]=="Portugal","gw_codes"]=235

vdem_s.loc[vdem_s["country"]=="Qatar","gw_codes"]=694

vdem_s.loc[vdem_s["country"]=="Romania","gw_codes"]=360
vdem_s.loc[vdem_s["country"]=="Russia","gw_codes"]=365
vdem_s.loc[vdem_s["country"]=="Rwanda","gw_codes"]=517
vdem_s.loc[vdem_s["country"]=="Republic of the Congo","gw_codes"]=484

vdem_s.loc[vdem_s["country"]=="Sao Tome and Principe","gw_codes"]=403
vdem_s.loc[vdem_s["country"]=="Saudi Arabia","gw_codes"]=670
vdem_s.loc[vdem_s["country"]=="Senegal","gw_codes"]=433
vdem_s.loc[vdem_s["country"]=="Serbia","gw_codes"]=340
vdem_s.loc[vdem_s["country"]=="Seychelles","gw_codes"]=591
vdem_s.loc[vdem_s["country"]=="Sierra Leone","gw_codes"]=451
vdem_s.loc[vdem_s["country"]=="Singapore","gw_codes"]=830
vdem_s.loc[vdem_s["country"]=="Slovakia","gw_codes"]=317
vdem_s.loc[vdem_s["country"]=="Slovenia","gw_codes"]=349
vdem_s.loc[vdem_s["country"]=="Solomon Islands","gw_codes"]=940
vdem_s.loc[vdem_s["country"]=="Somalia","gw_codes"]=520
vdem_s.loc[vdem_s["country"]=="South Africa","gw_codes"]=560
vdem_s.loc[vdem_s["country"]=="South Korea","gw_codes"]=732
vdem_s.loc[vdem_s["country"]=="South Sudan","gw_codes"]=626
vdem_s.loc[vdem_s["country"]=="Spain","gw_codes"]=230
vdem_s.loc[vdem_s["country"]=="Sri Lanka","gw_codes"]=780
vdem_s.loc[vdem_s["country"]=="Sudan","gw_codes"]=625
vdem_s.loc[vdem_s["country"]=="Suriname","gw_codes"]=115
vdem_s.loc[vdem_s["country"]=="Sweden","gw_codes"]=380
vdem_s.loc[vdem_s["country"]=="Switzerland","gw_codes"]=225
vdem_s.loc[vdem_s["country"]=="Syria","gw_codes"]=652

vdem_s.loc[vdem_s["country"]=="Taiwan","gw_codes"]=713
vdem_s.loc[vdem_s["country"]=="Tajikistan","gw_codes"]=702
vdem_s.loc[vdem_s["country"]=="Tanzania","gw_codes"]=510
vdem_s.loc[vdem_s["country"]=="Thailand","gw_codes"]=800
vdem_s.loc[vdem_s["country"]=="Togo","gw_codes"]=461
vdem_s.loc[vdem_s["country"]=="Trinidad and Tobago","gw_codes"]=52
vdem_s.loc[vdem_s["country"]=="Tunisia","gw_codes"]=616
vdem_s.loc[vdem_s["country"]=="Türkiye","gw_codes"]=640
vdem_s.loc[vdem_s["country"]=="Turkmenistan","gw_codes"]=701

vdem_s.loc[vdem_s["country"]=="Uganda","gw_codes"]=500
vdem_s.loc[vdem_s["country"]=="Ukraine","gw_codes"]=369
vdem_s.loc[vdem_s["country"]=="United Arab Emirates","gw_codes"]=696
vdem_s.loc[vdem_s["country"]=="United Kingdom","gw_codes"]=200
vdem_s.loc[vdem_s["country"]=="United States of America","gw_codes"]=2
vdem_s.loc[vdem_s["country"]=="Uruguay","gw_codes"]=165
vdem_s.loc[vdem_s["country"]=="Uzbekistan","gw_codes"]=704

vdem_s.loc[vdem_s["country"]=="Vanuatu","gw_codes"]=935
vdem_s.loc[vdem_s["country"]=="Venezuela","gw_codes"]=101
vdem_s.loc[vdem_s["country"]=="Vietnam","gw_codes"]=816

vdem_s.loc[vdem_s["country"]=="Yemen","gw_codes"]=678
vdem_s.loc[vdem_s["country"]=="Zambia","gw_codes"]=551
vdem_s.loc[vdem_s["country"]=="Zimbabwe","gw_codes"]=552

vdem_s.loc[vdem_s["country"]=="Eswatini","gw_codes"]=572

vdem_s=vdem_s.loc[vdem_s["gw_codes"]!=999999]

### drop countries which are missing ###

base=df[["year","country","gw_codes"]].drop_duplicates(subset=["year","country"]).reset_index(drop=True)
base=pd.merge(left=base,right=vdem_s[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_neopat","v2x_civlib","v2x_clphy","v2x_corr","v2x_rule"]],on=["year","gw_codes"],how="left")
base[base['v2x_libdem'].isna()].country.unique()

missing=['Andorra', 'Antigua and Barbuda', 'Bahamas', 'Belize', 'Brunei',
       'Dominica', 'Grenada', 'Kiribati', 'Liechtenstein',
       'Marshall Islands', 'Micronesia', 'Monaco', 'Nauru', 'Palau',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Tonga',
       'Tuvalu']

base = base[~base['country'].isin(missing)]
df = df[~df['country'].isin(missing)]

# Merge 
df=pd.merge(df,base[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_neopat","v2x_civlib","v2x_clphy","v2x_corr","v2x_rule"]],on=["year","gw_codes"],how="left")

                            ##################
                            ### World Bank ###
                            ##################

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
            'Taiwan': [713, 'TWN'],
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

c_codes=pd.DataFrame.from_dict(c_codes,orient='index')
c_codes = c_codes.reset_index()
c_codes.columns = ['country','gw_codes','iso_alpha3']

wdi = pd.DataFrame()
feat_dev = ["NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "SP.POP.TOTL", # Population size
            ]

# loop through each year and get data
for i in list(range(1989, 2024, 1)):
    print(i)
    wdi_s = wb.data.DataFrame(feat_dev, list(c_codes.iso_alpha3), [i])
    wdi_s.reset_index(inplace=True)
    wdi_s["year"] = i
    wdi = pd.concat([wdi, wdi_s], ignore_index=True)  # merge for each year

wdi = pd.merge(wdi,c_codes[['gw_codes','iso_alpha3']],how='left',left_on=['economy'],right_on=['iso_alpha3'])

# A. GDP per capita (current US$)

base=df[["year","country","gw_codes"]].drop_duplicates(subset=["year","country"]).reset_index(drop=True)
base=pd.merge(left=base,right=wdi[["year","gw_codes","NY.GDP.PCAP.CD","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
#base = base[~base['country'].isin(["North Korea","Taiwan","Venezuela"])] 

### Simple ###
base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.PCAP.CD"])

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PCAP.CD"].loc[base["country"]==c])
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["NY.GDP.PCAP.CD"].loc[base_imp_final["country"]==c])
    axs[0].set_title(c)
    plt.show()

df=pd.merge(df,base_imp_final[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")

# B. Population size

base=df[["year","country","gw_codes"]].drop_duplicates(subset=["year","country"]).reset_index(drop=True)
base=pd.merge(left=base,right=wdi[["year","gw_codes","NY.GDP.PCAP.CD","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")

### Simple ###
base_imp_final=linear_imp_grouped(base,"country",["SP.POP.TOTL"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.POP.TOTL"])

# Validate
for c in base.country.unique():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.TOTL"].loc[base["country"]==c])
    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["SP.POP.TOTL"].loc[base_imp_final["country"]==c])
    axs[0].set_title(c)
    plt.show()

# Merge
df=pd.merge(df,base_imp_final[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
df = df[~df['country'].isin(["North Korea","Taiwan","Venezuela"])]

                                #############
                                ### REIGN ###
                                #############

# Load data
#reign = pd.read_csv("data/reign_8_21.csv", encoding='latin1') 

# Add dates 
#reign['dd'] = pd.to_datetime(reign['month'].astype(str)+reign['year'].astype(str),format='%m%Y').dt.to_period('M')
#reign["dd"]=reign["dd"].astype(str)

# Subset
#reign = reign[["country",
#               "dd",
#               "year",
#               "tenure_months", # months that a leader has been in power
#               "anticipation", # there is an election for the de facto leadership position coming within the next six-months.
#               "lastelection", # time since the last election
#                  ]]

#reign["gw_codes"]=999999
#reign.loc[reign["country"]=="Afghanistan","gw_codes"]=700
#reign.loc[reign["country"]=="Albania","gw_codes"]=339
#reign.loc[reign["country"]=="Algeria","gw_codes"]=615
#reign.loc[reign["country"]=="Andorra","gw_codes"]=232
#reign.loc[reign["country"]=="Angola","gw_codes"]=540
#reign.loc[reign["country"]=="Antigua and Barbuda","gw_codes"]=58
#reign.loc[reign["country"]=="Argentina","gw_codes"]=160
#reign.loc[reign["country"]=="Armenia","gw_codes"]=371
#reign.loc[reign["country"]=="Australia","gw_codes"]=900
#reign.loc[reign["country"]=="Austria","gw_codes"]=305
#reign.loc[reign["country"]=="Azerbaijan","gw_codes"]=373

#reign.loc[reign["country"]=="Bahamas","gw_codes"]=31
#reign.loc[reign["country"]=="Bahrain","gw_codes"]=692
#reign.loc[reign["country"]=="Bangladesh","gw_codes"]=771
#reign.loc[reign["country"]=="Barbados","gw_codes"]=53
#reign.loc[reign["country"]=="Belarus","gw_codes"]=370
#reign.loc[reign["country"]=="Belgium","gw_codes"]=211
#reign.loc[reign["country"]=="Belize","gw_codes"]=80
#reign.loc[reign["country"]=="Benin","gw_codes"]=434
#reign.loc[reign["country"]=="Bhutan","gw_codes"]=760
#reign.loc[reign["country"]=="Bolivia","gw_codes"]=145
#reign.loc[reign["country"]=="Bosnia","gw_codes"]=346
#reign.loc[reign["country"]=="Botswana","gw_codes"]=571
#reign.loc[reign["country"]=="Brazil","gw_codes"]=140
#reign.loc[reign["country"]=="Brunei","gw_codes"]=835
#reign.loc[reign["country"]=="Bulgaria","gw_codes"]=355
#reign.loc[reign["country"]=="Burkina Faso","gw_codes"]=439
#reign.loc[reign["country"]=="Burundi","gw_codes"]=516

#reign.loc[reign["country"]=="Cape Verde","gw_codes"]=402
#reign.loc[reign["country"]=="Cambodia","gw_codes"]=811
#reign.loc[reign["country"]=="Cameroon","gw_codes"]=471
#reign.loc[reign["country"]=="Canada","gw_codes"]=20
#reign.loc[reign["country"]=="Cen African Rep","gw_codes"]=482
#reign.loc[reign["country"]=="Chad","gw_codes"]=483
#reign.loc[reign["country"]=="Chile","gw_codes"]=155
#reign.loc[reign["country"]=="China","gw_codes"]=710
#reign.loc[reign["country"]=="Colombia","gw_codes"]=100
#reign.loc[reign["country"]=="Comoros","gw_codes"]=581
#reign.loc[reign["country"]=="Costa Rica","gw_codes"]=94
#reign.loc[reign["country"]=="Croatia","gw_codes"]=344
#reign.loc[reign["country"]=="Cuba","gw_codes"]=40
#reign.loc[reign["country"]=="Cyprus","gw_codes"]=352
#reign.loc[reign["country"]=="Czech Rep","gw_codes"]=316
#reign.loc[reign["country"]=="Congo-Brz","gw_codes"]=484
#reign.loc[reign["country"]=="Congo/Zaire","gw_codes"]=490
#reign.loc[reign["country"]=="Czechoslovakia","gw_codes"]=315

#reign.loc[reign["country"]=="Denmark","gw_codes"]=390
#reign.loc[reign["country"]=="Djibouti","gw_codes"]=522
#reign.loc[reign["country"]=="Dominica","gw_codes"]=54
#reign.loc[reign["country"]=="Dominican Rep","gw_codes"]=42

#reign.loc[reign["country"]=="East Timor","gw_codes"]=860
#reign.loc[reign["country"]=="Ecuador","gw_codes"]=130
#reign.loc[reign["country"]=="Egypt","gw_codes"]=651
#reign.loc[reign["country"]=="El Salvador","gw_codes"]=92
#reign.loc[reign["country"]=="Equatorial Guinea","gw_codes"]=411
#reign.loc[reign["country"]=="Eritrea","gw_codes"]=531
#reign.loc[reign["country"]=="Estonia","gw_codes"]=366
#reign.loc[reign["country"]=="Ethiopia","gw_codes"]=530

#reign.loc[reign["country"]=="Fiji","gw_codes"]=950
#reign.loc[reign["country"]=="Finland","gw_codes"]=375
#reign.loc[reign["country"]=="France","gw_codes"]=220

#reign.loc[reign["country"]=="Gabon","gw_codes"]=481
#reign.loc[reign["country"]=="Gambia","gw_codes"]=420
#reign.loc[reign["country"]=="Georgia","gw_codes"]=372
#reign.loc[reign["country"]=="Germany","gw_codes"]=260
#reign.loc[reign["country"]=="Germany East","gw_codes"]=265
#reign.loc[reign["country"]=="Ghana","gw_codes"]=452
#reign.loc[reign["country"]=="Greece","gw_codes"]=350
#reign.loc[reign["country"]=="Grenada","gw_codes"]=55
#reign.loc[reign["country"]=="Guatemala","gw_codes"]=90
#reign.loc[reign["country"]=="Guinea","gw_codes"]=438
#reign.loc[reign["country"]=="Guinea Bissau","gw_codes"]=404
#reign.loc[reign["country"]=="Guyana","gw_codes"]=110

#reign.loc[reign["country"]=="Haiti","gw_codes"]=41
#reign.loc[reign["country"]=="Honduras","gw_codes"]=91
#reign.loc[reign["country"]=="Hungary","gw_codes"]=310

#reign.loc[reign["country"]=="Iceland","gw_codes"]=395
#reign.loc[reign["country"]=="India","gw_codes"]=750
#reign.loc[reign["country"]=="Indonesia","gw_codes"]=850
#reign.loc[reign["country"]=="Iran","gw_codes"]=630
#reign.loc[reign["country"]=="Iraq","gw_codes"]=645
#reign.loc[reign["country"]=="Ireland","gw_codes"]=205
#reign.loc[reign["country"]=="Israel","gw_codes"]=666
#reign.loc[reign["country"]=="Italy","gw_codes"]=325
#reign.loc[reign["country"]=="Ivory Coast","gw_codes"]=437

#reign.loc[reign["country"]=="Jamaica","gw_codes"]=51
#reign.loc[reign["country"]=="Japan","gw_codes"]=740
#reign.loc[reign["country"]=="Jordan","gw_codes"]=663

#reign.loc[reign["country"]=="Kazakhstan","gw_codes"]=705
#reign.loc[reign["country"]=="Kenya","gw_codes"]=501
#reign.loc[reign["country"]=="Korea North","gw_codes"]=731
#reign.loc[reign["country"]=="Korea South","gw_codes"]=732
#reign.loc[reign["country"]=="Kosovo","gw_codes"]=347
#reign.loc[reign["country"]=="Kuwait","gw_codes"]=690
#reign.loc[reign["country"]=="Kyrgyzstan","gw_codes"]=703

#reign.loc[reign["country"]=="Laos","gw_codes"]=812
#reign.loc[reign["country"]=="Latvia","gw_codes"]=367
#reign.loc[reign["country"]=="Lebanon","gw_codes"]=660
#reign.loc[reign["country"]=="Lesotho","gw_codes"]=570
#reign.loc[reign["country"]=="Liberia","gw_codes"]=450
#reign.loc[reign["country"]=="Libya","gw_codes"]=620
#reign.loc[reign["country"]=="Lithuania","gw_codes"]=368
#reign.loc[reign["country"]=="Luxembourg","gw_codes"]=212

#reign.loc[reign["country"]=="Macedonia","gw_codes"]=343
#reign.loc[reign["country"]=="Madagascar","gw_codes"]=580
#reign.loc[reign["country"]=="Malawi","gw_codes"]=553
#reign.loc[reign["country"]=="Malaysia","gw_codes"]=820
#reign.loc[reign["country"]=="Maldives","gw_codes"]=781
#reign.loc[reign["country"]=="Mali","gw_codes"]=432
#reign.loc[reign["country"]=="Malta","gw_codes"]=338

#reign.loc[reign["country"]=="Mauritius","gw_codes"]=590
#reign.loc[reign["country"]=="Mauritania","gw_codes"]=435
#reign.loc[reign["country"]=="Mexico","gw_codes"]=70
#reign.loc[reign["country"]=="Moldova","gw_codes"]=359
#reign.loc[reign["country"]=="Mongolia","gw_codes"]=712
#reign.loc[reign["country"]=="Montenegro","gw_codes"]=341
#reign.loc[reign["country"]=="Morocco","gw_codes"]=600
#reign.loc[reign["country"]=="Mozambique","gw_codes"]=541
#reign.loc[reign["country"]=="Myanmar","gw_codes"]=775

#reign.loc[reign["country"]=="Namibia","gw_codes"]=565
#reign.loc[reign["country"]=="Nepal","gw_codes"]=790
#reign.loc[reign["country"]=="Netherlands","gw_codes"]=210
#reign.loc[reign["country"]=="New Zealand","gw_codes"]=920
#reign.loc[reign["country"]=="Nicaragua","gw_codes"]=93
#reign.loc[reign["country"]=="Niger","gw_codes"]=436
#reign.loc[reign["country"]=="Nigeria","gw_codes"]=475
#reign.loc[reign["country"]=="Norway","gw_codes"]=385

#reign.loc[reign["country"]=="Oman","gw_codes"]=698

#reign.loc[reign["country"]=="Pakistan","gw_codes"]=770
#reign.loc[reign["country"]=="Panama","gw_codes"]=95
#reign.loc[reign["country"]=="Papua New Guinea","gw_codes"]=910
#reign.loc[reign["country"]=="Paraguay","gw_codes"]=150
#reign.loc[reign["country"]=="Peru","gw_codes"]=135
#reign.loc[reign["country"]=="Philippines","gw_codes"]=840
#reign.loc[reign["country"]=="Poland","gw_codes"]=290
#reign.loc[reign["country"]=="Portugal","gw_codes"]=235

#reign.loc[reign["country"]=="Qatar","gw_codes"]=694#

#reign.loc[reign["country"]=="Romania","gw_codes"]=360
#reign.loc[reign["country"]=="Russia","gw_codes"]=365
#reign.loc[reign["country"]=="Rwanda","gw_codes"]=517

#reign.loc[reign["country"]=="Sao Tome and Principe","gw_codes"]=403
#reign.loc[reign["country"]=="Saudi Arabia","gw_codes"]=670
#reign.loc[reign["country"]=="Senegal","gw_codes"]=433
#reign.loc[reign["country"]=="Serbia","gw_codes"]=340
#reign.loc[reign["country"]=="Seychelles","gw_codes"]=591
#reign.loc[reign["country"]=="Sierra Leone","gw_codes"]=451
#reign.loc[reign["country"]=="Singapore","gw_codes"]=830
#reign.loc[reign["country"]=="Slovakia","gw_codes"]=317
#reign.loc[reign["country"]=="Slovenia","gw_codes"]=349
#reign.loc[reign["country"]=="Solomon Islands","gw_codes"]=940
#reign.loc[reign["country"]=="Somalia","gw_codes"]=520
#reign.loc[reign["country"]=="South Africa","gw_codes"]=560
#reign.loc[reign["country"]=="South Sudan","gw_codes"]=626
#reign.loc[reign["country"]=="Spain","gw_codes"]=230
#reign.loc[reign["country"]=="Sri Lanka","gw_codes"]=780
#reign.loc[reign["country"]=="Sudan","gw_codes"]=625
#reign.loc[reign["country"]=="Suriname","gw_codes"]=115
#reign.loc[reign["country"]=="Sweden","gw_codes"]=380
#reign.loc[reign["country"]=="Switzerland","gw_codes"]=225
#reign.loc[reign["country"]=="Syria","gw_codes"]=652

#reign.loc[reign["country"]=="Tajikistan","gw_codes"]=702
#reign.loc[reign["country"]=="Tanzania","gw_codes"]=510
#reign.loc[reign["country"]=="Thailand","gw_codes"]=800
#reign.loc[reign["country"]=="Togo","gw_codes"]=461
#reign.loc[reign["country"]=="Trinidad and Tobago","gw_codes"]=52
#reign.loc[reign["country"]=="Tunisia","gw_codes"]=616
#reign.loc[reign["country"]=="Turkey","gw_codes"]=640
#reign.loc[reign["country"]=="Turkmenistan","gw_codes"]=701

#reign.loc[reign["country"]=="Uganda","gw_codes"]=500
#reign.loc[reign["country"]=="Ukraine","gw_codes"]=369
#reign.loc[reign["country"]=="United Arab Emirates","gw_codes"]=696
#reign.loc[reign["country"]=="UKG","gw_codes"]=200
#reign.loc[reign["country"]=="USA","gw_codes"]=2
#reign.loc[reign["country"]=="Uruguay","gw_codes"]=165
#reign.loc[reign["country"]=="Uzbekistan","gw_codes"]=704

#reign.loc[reign["country"]=="Vanuatu","gw_codes"]=935
#reign.loc[reign["country"]=="Venezuela","gw_codes"]=101
#reign.loc[reign["country"]=="Vietnam","gw_codes"]=816

#reign.loc[reign["country"]=="Yemen","gw_codes"]=678
#reign.loc[reign["country"]=="Zambia","gw_codes"]=551
#reign.loc[reign["country"]=="Zimbabwe","gw_codes"]=552

#reign.loc[reign["country"]=="Swaziland","gw_codes"]=572

#reign.loc[reign["country"]=="St Lucia","gw_codes"]=56
#reign.loc[reign["country"]=="St Vincent","gw_codes"]=57
#reign.loc[reign["country"]=="St Kitts and Nevis","gw_codes"]=60
#reign.loc[reign["country"]=="Monaco","gw_codes"]=221
#reign.loc[reign["country"]=="Liechtenstein","gw_codes"]=223
#reign.loc[reign["country"]=="San Marino","gw_codes"]=331
#reign.loc[reign["country"]=="Yugoslavia","gw_codes"]=345
#reign.loc[reign["country"]=="Yemen South","gw_codes"]=680
#reign.loc[reign["country"]=="Vietnam South","gw_codes"]=817
#reign.loc[reign["country"]=="Kiribati","gw_codes"]=970
#reign.loc[reign["country"]=="Tuvalu","gw_codes"]=973
#reign.loc[reign["country"]=="Tonga","gw_codes"]=972
#reign.loc[reign["country"]=="Nauru","gw_codes"]=971
#reign.loc[reign["country"]=="Marshall Islands","gw_codes"]=983
#reign.loc[reign["country"]=="Palau","gw_codes"]=986
#reign.loc[reign["country"]=="Micronesia","gw_codes"]=987
#reign.loc[reign["country"]=="Samoa","gw_codes"]=990

#reign=reign.loc[reign["gw_codes"]!=999999]

# Remove duplicates in REIGN 
#reign.drop_duplicates(keep='first', inplace=True,subset=['gw_codes',"dd"])
    
# A. Months in power

#base=df[["dd","country","gw_codes"]]
#base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","anticipation","lastelection"]],on=["dd","gw_codes"],how="left")

### Simple ###
#base_imp_final=linear_imp_grouped(base,"country",["tenure_months"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["tenure_months"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["tenure_months"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge 
#df=pd.merge(df,base_imp_final[["dd","gw_codes","tenure_months"]],on=["dd","gw_codes"],how="left")

# B. Elections anticipated

#base=df[["dd","country","gw_codes"]]
#base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","anticipation","lastelection"]],on=["dd","gw_codes"],how="left")

### Multiple ###

#base_imp=imp_opti(base,"country",["anticipation"],vars_add=["tenure_months","lastelection"],max_iter=10)

# Calibrate 
#base_imp_calib=calibrate_imp(base_imp, "country", "anticipation")

### Simple ###
#base_imp_final=linear_imp_grouped(base,"country",["anticipation"])

# Merge
#base_imp_final['anticipation'] = base_imp_final['anticipation'].fillna(base_imp_calib['anticipation'])
#base_imp_final=base_imp_final.sort_values(by=["country","dd"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["anticipation"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["anticipation"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge 
#df=pd.merge(df,base_imp_final[["dd","gw_codes","anticipation"]],on=["dd","gw_codes"],how="left")

# C. Months since elections

#base=df[["dd","country","gw_codes"]]
#base=pd.merge(left=base,right=reign[["dd","gw_codes","tenure_months","anticipation","lastelection"]],on=["dd","gw_codes"],how="left")

### Multiple ###

#base_imp=imp_opti(base,"country",["lastelection"],vars_add=["tenure_months","anticipation"],max_iter=10)

# Calibrate 
#base_imp_calib=calibrate_imp(base_imp, "country", "lastelection")

### Simple ###
#base_imp_final=linear_imp_grouped(base,"country",["lastelection"])

# Merge
#base_imp_final['lastelection'] = base_imp_final['lastelection'].fillna(base_imp_calib['lastelection'])
#base_imp_final=base_imp_final.sort_values(by=["country","dd"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["dd"].loc[base["country"]==c], base["lastelection"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["dd"].loc[base_imp_final["country"]==c], base_imp_final["lastelection"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge 
#df=pd.merge(df,base_imp_final[["dd","gw_codes","lastelection"]],on=["dd","gw_codes"],how="left")

# Save
df.isnull().any()
df.to_csv("data/df.csv")  



