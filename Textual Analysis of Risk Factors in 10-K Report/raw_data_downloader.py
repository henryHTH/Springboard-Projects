# Importing built-in libraries (no need to install these)
import re
import os
from time import gmtime, strftime, localtime
from datetime import datetime, timedelta
import unicodedata

# Importing libraries you need to install
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import bs4 as bs
from lxml import html
from tqdm import tqdm

def RemoveTags(soup):
    
    '''
    Drops HTML tags, newlines and unicode text from
    filing text.
    
    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.
        
    Returns
    -------
    text : str
        Filing text.
        
    '''
    
    # Remove HTML tags with get_text
    text = soup.get_text()
    
    # Remove non acii characters
    text = re.sub('\n|\t|\r|\x0b|\x0c', ' ', text)
    
    # Replace unicode characters with their
    # "normal" representations
    text = unicodedata.normalize('NFKD', text)
    
    return text


def extract_risk_factors(text):

    '''
    Extrack Item 1A. Risk Factors part by regular expression from 10-K report.
    
    Parameters
    ----------
    text : string
        Parsed and cleaned text from BeautifulSoup.
        
    Returns
    -------
    text : str
        Item 1A. Risk Factors text.
        
    '''

    start,end = -1,-1
    look={" see ", " refer to ", " included in "," contained in ",' in '}
    name_1a = re.compile('Item *1A\. *Risk *Factors', re.IGNORECASE)
    name_1b = re.compile('Item *1B\. *Unresolved *Staff *Comments', re.IGNORECASE)
    
    first_flag = sum(1 for _ in name_1a.finditer(text,re.IGNORECASE))!=1                
    for m in name_1a.finditer(text,re.IGNORECASE):
        if first_flag == False:
            substring = text[m.start()-20:m.start()]
            if not any(s in substring for s in look):
                if ('Item' in re.split('[\W]+',text[m.end():m.end()+40])) | ('Item' in re.split('[\W]+',text[m.start()-40:m.start()])):
                    continue
                start = m.span()[0]
                #print(start)
                break
        first_flag = False
    

    first_flag = sum(1 for _ in name_1b.finditer(text,re.IGNORECASE))!=1
    for m in name_1b.finditer(text,re.IGNORECASE):
        if first_flag == False:
            substring = text[m.start()-20:m.start()]
            if not any(s in substring for s in look):   
                end = m.span()[0]
                if end < start:
                    continue
                #print(end)
                break
        first_flag = False
    
    #print(start,end)
    if (start != -1) & (end != -1):
        return text[start:end]
    else:
        return 
    
    
def Extract_Risk_Factors(soup):

	'''
    Extrack Item 1A. Risk Factors part by find the anchor in the web where 10-K report lies.
    
    Parameters
    ----------
    soup : beautifulsoup object
        Parsed from BeautifulSoup.
        
    Returns
    -------
    text : str
        Item 1A. Risk Factors text.
    or fundtion extract_risk_factors, if text is none 
    '''

	name_1a,name_1b = '',''
	find_1a = False
	find_1b = False
	href_flag = False
	if len(soup.find_all('a')) != 0: 
	    for s in soup.find_all('a'):
	        #p_1a = re.compile('Item.1A', re.IGNORECASE)
	        #p_1b = re.compile('Item.1B', re.IGNORECASE)
	        text = unicodedata.normalize('NFKD', s.text)
	        text = re.sub('\n|\t|\r|\x0b|\x0c', ' ', text)
	        if not find_1a:
	            if re.match('Item *1A|Risk *Factors|Item *1A\. *Risk *Factors',text.strip(), re.IGNORECASE):
	            #(re.match('Item 1A',text.strip())) or (re.match('Risk Factors',text.strip(), re.IGNORECASE)) or (re.match('Item *1A\. *Risk *Factors',text.strip(), re.IGNORECASE)) :
	            	if hasattr(s.attr,'href'):
	                	name_1a = s.attrs['href'][1:]
	                	find_1a = True
	        if not find_1b:
	            if re.match('Item *1B|Unresolved *Staff *Comments|Item *1B\. *Unresolved *Staff *Comments',text.strip(), re.IGNORECASE):
	            #(re.match('Item 1B',text.strip())) or (re.match('Unresolved Staff Comments',text.strip(), re.IGNORECASE)) or (re.match('Item *1B\. *Unresolved *Staff *Comments',text.strip(), re.IGNORECASE)):
	            	if hasattr(s.attr,'href'):
	                	name_1b = s.attrs['href'][1:]
	                	find_1b = True
	        if find_1a & find_1b: 
	            href_flag = True
	            break

	    if find_1a & find_1b:
	        for name in ['name','id']:
	            a = str(soup).find(f'<a {name}="{name_1a}">',re.IGNORECASE)
	            b = str(soup).find(f'<a {name}="{name_1b}">',re.IGNORECASE)
	            if (a==-1) & (b==-1):
	                a = str(soup).lower().find(f'<a {name}="{name_1a}">'.lower(),re.IGNORECASE)
	                b = str(soup).lower().find(f'<a {name}="{name_1b}">'.lower(),re.IGNORECASE)
	            if (a!=-1) & (b!=-1):
	                risk_file = str(soup)[a:b]
	                cleansoup = bs.BeautifulSoup(risk_file, "lxml")
	                # Drops HTML tags, newlines and unicode text from filing text.
	                risk_text = RemoveTags(cleansoup)
	                #print(risk_text)
	                return risk_text
	            else:
	                continue
	        if (a==-1) & (b==-1):
	            #print('Cant find a match')
	            return extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))
	    else:
	        #print('Cant find a match')
	        return  extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))
	else:
	    #print('Cant find a match')
	    return extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))


def MapTickerToCik(tickers):
    url = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
    cik_re = re.compile(r'.*CIK=(\d{10}).*')

    cik_dict = {}
    for ticker in tqdm(tickers): # Use tqdm lib for progress bar
        results = cik_re.findall(requests.get(url.format(ticker)).text)
        if len(results):
            cik_dict[str(ticker).lower()] = str(results[0])
    
    return cik_dict

def Scrape10K(browse_url_base, filing_url_base, doc_url_base, cik, log_file_name):
    
    '''
    Scrapes all 10-Ks and 10-K405s for a particular 
    CIK from EDGAR.
    
    Parameters
    ----------
    browse_url_base : str
        Base URL for browsing EDGAR.
    filing_url_base : str
        Base URL for filings listings on EDGAR.
    doc_url_base : str
        Base URL for one filing's document tables
        page on EDGAR.
    cik : str
        Central Index Key.
    log_file_name : str
        Name of the log file (should be a .txt file).
        
    Returns
    -------
    None.
    
    '''
    
    # Check if we've already scraped this CIK
    try:
        os.mkdir(cik)
    except OSError:
        print("Already scraped CIK", cik)
        return
    
    # If we haven't, go into the directory for that CIK
    os.chdir(cik)
    
    print('Scraping CIK', cik)
    
    # Request list of 10-K filings
    res = requests.get(browse_url_base % cik)
    
    # If the request failed, log the failure and exit
    if res.status_code != 200:
        os.chdir('..')
        os.rmdir(cik) # remove empty dir
        text = "Request failed with error code " + str(res.status_code) + \
               "\nFailed URL: " + (browse_url_base % cik) + '\n'
        WriteLogFile(log_file_name, text)
        return

    # If the request doesn't fail, continue...
    
    # Parse the response HTML using BeautifulSoup
    soup = bs.BeautifulSoup(res.text, "lxml")

    # Extract all tables from the response
    html_tables = soup.find_all('table')
    
    # Check that the table we're looking for exists
    # If it doesn't, exit
    if len(html_tables)<3:
        os.chdir('..')
        return
    
    # Parse the Filings table
    filings_table = pd.read_html(str(html_tables[2]), header=0)[0]
    filings_table['Filings'] = [str(x) for x in filings_table['Filings']]

    # Get only 10-K and 10-K405 document filings
    filings_table = filings_table[(filings_table['Filings'] == '10-K')]
    filings_table = filings_table[(filings_table['Filing Date']<'2020-01-01') & (filings_table['Filing Date']>'2006-01-01')]


    # If filings table doesn't have any
    # 10-Ks, exit
    if len(filings_table)==0:
        os.chdir('..')
        return
    
    # Get accession number for each 10-K filing
    filings_table['Acc_No'] = [x.replace('\xa0',' ')
                               .split('Acc-no: ')[1]
                               .split(' ')[0] for x in filings_table['Description']]

    # Iterate through each filing and 
    # scrape the corresponding document...
    for index, row in filings_table.iterrows():
        
        # Get the accession number for the filing
        acc_no = str(row['Acc_No'])
        #date = str(row['Filing Date'])
        # Navigate to the page for the filing
        docs_page = requests.get(filing_url_base % (cik, acc_no))
        
        # If request fails, log the failure
        # and skip to the next filing
        if docs_page.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(docs_page.status_code) + \
                   "\nFailed URL: " + (filing_url_base % (cik, acc_no, date)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue

        # If request succeeds, keep going...
        
        # Parse the table of documents for the filing
        docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
        docs_html_tables = docs_page_soup.find_all('table')
        if len(docs_html_tables)==0:
            continue
        docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
        docs_table['Type'] = [str(x) for x in docs_table['Type']]
        
        # Get the 10-K and 10-K405 entries for the filing
        docs_table = docs_table[(docs_table['Type'] == '10-K')]
        
        # If there aren't any 10-K or 10-K405 entries,
        # skip to the next filing
        if len(docs_table)==0:
            continue
        # If there are 10-K or 10-K405 entries,
        # grab the first document
        elif len(docs_table)>0:
            docs_table = docs_table.iloc[0]
        
        docname = docs_table['Document']
        if 'iXBRL' in docname:
        	docname = docname.split(' ')[0]

        
        date = [x.text for x in docs_page_soup.find_all('div',{"class": "info"})][-1]
        if type(date) != str:
        	date = 'not_available'
        # If that first entry is unavailable,
        # log the failure and exit
        if str(docname) == 'nan':
            os.chdir('..')
            text = 'File with CIK: %s and Acc_No (date: %s): %s is unavailable' % (cik, date, acc_no) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue       
        
        # If it is available, continue...
        
        # Request the file
        file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))
        
        # If the request fails, log the failure and exit
        if file.status_code != 200:
            os.chdir('..')
            text = "Request failed with error code " + str(file.status_code) + \
                   "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue
        
        # If it succeeds, keep going...
        # Sparse 10-K report 
        file_soup = bs.BeautifulSoup(file.text, 'lxml')

        # If flag is true, means extract_risk_factors function failed to get correct data, need further analysis
        #if need_file is True:
        #    print(file.url)
        # Extract Item 1A Risk Factors 
        risk_file = Extract_Risk_Factors(file_soup)
        
        if risk_file is None:
            os.chdir('..')
            text = 'Risk Factors section with CIK: %s and Acc_No (date: %s): %s is unavailable' % (cik, date, acc_no) + '\n'
            WriteLogFile(log_file_name, text)
            os.chdir(cik)
            continue
        
        
        # Save the file in appropriate format
        if '.txt' in docname:
            # Save text as TXT
            #date = str(row['Filing Date'])
            filename = cik + '_' + date + '.txt'
            html_file = open(filename, 'a')
            html_file.write(risk_file)
            html_file.close()
        else:
            # Save text tagged with HTML
            #date = str(row['Filing Date'])
            filename = cik + '_' + date + '_html.txt'
            html_file = open(filename, 'a')
            html_file.write(risk_file)
            html_file.close()
        
    # Move back to the main 10-K directory
    os.chdir('..')
        
    return


def WriteLogFile(log_file_name, text):
    
    '''
    Helper function.
    Writes a log file with all notes and
    error messages from a scraping "session".
    
    Parameters
    ----------
    log_file_name : str
        Name of the log file (should be a .txt file).
    text : str
        Text to write to the log file.
        
    Returns
    -------
    None.
    
    '''
    
    with open(log_file_name, "a") as log_file:
        log_file.write(text)

    return


def removeEmptyFolders(path, removeRoot=True): 
    """
    Helper Function.
    Remove all empty folders in current direction 
    
    Parameters
    ----------
    path : str
        The direction to remove all empty folders.
    removeRoot : Boolean 
        Whether to delete all empty subfolders, default is True
        
    Returns
    -------
    None.
    
    """
    
    if not os.path.isdir(path):
        return 
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)
    files = os.listdir(path)
    if (len(files) == 0) & removeRoot:
            print (f'Removing empty folder:{path}')
            os.rmdir(path)


def cik_list_download():	

	# Get lists of tickers from NASDAQ, NYSE, AMEX
	nasdaq_tickers = pd.read_csv('US-Stock-Symbols/companylist_nasdaq.csv')
	nyse_tickers = pd.read_csv('US-Stock-Symbols/companylist_nyse.csv')
	amex_tickers = pd.read_csv('US-Stock-Symbols/companylist_amex.csv')

	# Drop irrelevant cols
	nasdaq_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)
	nyse_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)
	amex_tickers.drop(labels='Unnamed: 8', axis='columns', inplace=True)

	# Create full list of tickers/names across all 3 exchanges
	tickers = list(set(list(nasdaq_tickers['Symbol']) + list(nyse_tickers['Symbol']) + list(amex_tickers['Symbol'])))
	cik_dict = MapTickerToCik(tickers)

	# Clean up the ticker-CIK mapping as a DataFrame
	ticker_cik_df = pd.DataFrame.from_dict(data=cik_dict, orient='index')
	ticker_cik_df.reset_index(inplace=True)
	ticker_cik_df.columns = ['ticker', 'cik']
	ticker_cik_df['cik'] = [str(cik) for cik in ticker_cik_df['cik']]

	# Keep first ticker alphabetically for duplicated CIKs
	ticker_cik_df = ticker_cik_df.sort_values(by='ticker')
	ticker_cik_df.drop_duplicates(subset='cik', keep='first', inplace=True)

	# Check that we've eliminated duplicate tickers/CIKs
	print("Number of ticker-cik pairings:", len(ticker_cik_df))
	print("Number of unique tickers:", len(set(ticker_cik_df['ticker'])))
	print("Number of unique CIKs:", len(set(ticker_cik_df['cik'])))

	return ticker_cik_df

if __name__ == '__main__':

	pathname = r'/Users/henry/Documents/study/Springboard/nlp_project'
	data_pathname = r'/Users/henry/Documents/study/Springboard/nlp_project/data'

	os.chdir(pathname)
	try:
		ticker_cik_df = pd.read_csv(r'US-Stock-Symbols/ticker_cik.csv').astype('str')
		ticker_cik_df['cik'] = ticker_cik_df['cik'].apply(lambda x: x.zfill(10))
	except FileNotFoundError:
		ticker_cik_df = cik_list_download()

	# Remove empty folders
	removeEmptyFolders(pathname, removeRoot=True)

	# Run the function to scrape 10-Ks
	'''
	# Define parameters
	browse_url_base_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K'
	filing_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
	doc_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

	# Set correct directory
	os.chdir(data_pathname)

	# Initialize log file
	# (log file name = the time we initiate scraping session)
	#time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
	time = strftime("%y%y-%m-%d,%H:%M:%S", localtime())
	log_file_name = 'log '+time+'.txt'
	with open(log_file_name, 'a') as log_file:
		log_file.write(time + '\n')
		log_file.close()

	# Iterate over CIKs and scrape 10-Ks
	for cik in tqdm(ticker_cik_df['cik']):
		Scrape10K(browse_url_base=browse_url_base_10k, 
	          filing_url_base=filing_url_base_10k, 
	          doc_url_base=doc_url_base_10k, 
	          cik=cik,
	          log_file_name=log_file_name)

	with open(log_file_name, 'a') as log_file:
		log_file.write(strftime("%y%y-%m-%d,%H:%M:%S", localtime()) + '\n')
		log_file.close()

	'''
# 0001069308, 0000934549, 0000353020, 0000874292, 0000922864 KeyError: 'href'
