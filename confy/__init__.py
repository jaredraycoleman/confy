import argparse
from itertools import chain
import logging
import pathlib
from datetime import datetime
from typing import Generator, List, Optional, Set, Tuple

import html2text
import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import fuzz

thisdir = pathlib.Path(__file__).parent.resolve()
base_url = "http://www.wikicfp.com/cfp/servlet/tool.search"

# set logging level to INFO
logging.basicConfig(level=logging.INFO)

def get_all_series() -> Generator[Tuple[str, str, str], None, None]:
    # loop over each letter in the alphabet
    for letter in "abcdefghijklmnopqrstuvwxyz".upper():
        res = requests.get(f"http://www.wikicfp.com/cfp/series?t=c&i={letter}")

        # iterate over all td tags in the html
        for td in BeautifulSoup(res.text, "html.parser").find_all("td"):
            td: BeautifulSoup
            # if td doesn't have a direct child a tag, continue
            if not td.find("a", recursive=False):
                continue
            # if the href of the a tag doesn't start with /cfp/program, continue
            if not td.find("a")["href"].startswith("/cfp/program"):
                continue
            acronym = td.find("a").text
            url = td.find("a")["href"]
            # get non a tag inner text
            name = td.text.replace(f"{acronym} - ", "").strip()

            # fix UTF-8 encoding
            name = name.encode("latin1").decode("utf8")

            yield acronym, name, url

def get_additional_series() -> Generator[Tuple[str, str, str], None, None]:
    yield "SIROCCO", "International Colloquium on Structural Information and Communication Complexity", "/cfp/servlet/event.showcfp?eventid=170703&copyownerid=178598"
    yield "BLOCKCHAIN", "International Conference on Blockchain", "/cfp/servlet/event.showcfp?eventid=171710&copyownerid=95447"

def get_series_cfps(skip: Set[str], do_series: bool = True, query: Optional[str] = None) -> Generator[Tuple[str, str, str, datetime, datetime, str, str, str], None, None]:
    """Get all CFPs for all series from WikiCFP

    Args:
        skip (Set[str]): Set of series to skip
        do_series (bool, optional): Whether to get CFPs for all series. Defaults to True.
        query (Optional[str], optional): Query to filter CFPs. Defaults to None.

    Yields:
        Generator[Tuple[str, str, str, datetime, datetime, str, str, str], None, None]: 
            (acronym, name, event, start, end, where, deadline, cfp)
    """
    funcs = []
    if do_series:
        funcs.append(get_all_series)
        funcs.append(get_additional_series)
    if query:
        funcs.append(lambda: search_wiki_cfp(query))
    
    for acronym, name, url in chain(*map(lambda f: f(), funcs)):
        # skip if acronym is in skip
        if acronym in skip:
            logging.debug(f"Skipping {acronym} - {name}")
            continue

        logging.info(f"Getting CFPs for {acronym} - {name}")

        res = requests.get(f"http://www.wikicfp.com{url}")
        cfp = BeautifulSoup(res.text, "html.parser").find("div", class_="cfp")
        cfp = html2text.html2text(str(cfp))
        cfp = cfp.encode("latin1", errors="ignore").decode("utf8", errors="ignore")

        # get tables
        tables = pd.read_html(res.text)

        when, where, deadline = None, None, None
        for table in tables:
            # check if the first row starts with [Event, When, Where, Deadline]
            cols = ["Event", "When", "Where", "Deadline"]
            first_row = table.iloc[0].tolist()
            if all(col in first_row for col in cols):
                # make first row header
                table.columns = first_row
                table = table.drop(0)

                # get the conference name
                name = table.iloc[0]["When"]
                # get the conference information
                when, where, deadline  = table.iloc[1, 1:].tolist()
                # parse when into start date and end date
                start, end = when.split(" - ")
                # parse deadline into deadline, removing additional dates in parentheses
                deadline = deadline.split("(")[0].strip()

            # check if first column is [Event, When, Where, Deadline]
            _cols = ["When", "Where", "Submission Deadline"]
            first_col = table.iloc[:, 0].tolist()
            if all(col in first_col for col in _cols):
                table = table.T

                first_row = table.iloc[0].tolist()
                
                # make first row header
                table.columns = first_row
                table = table.drop(0)

                # get the conference information
                when, where, deadline  = table[_cols].iloc[0].tolist()
                # parse when into start date and end date
                try:
                    start, end = when.split(" - ")
                except Exception:
                    start, end = None, None
                # parse deadline into deadline, removing additional dates in parentheses
                deadline = deadline.split("(")[0].strip()

        yield acronym, name, url, start, end, where, deadline, cfp

def search_wiki_cfp(query: List[List[str]]) -> Generator[Tuple[str, str, str], None, None]:
    query_flat = [item for sublist in query for item in sublist]
    query_str = " ".join(query_flat)

    params = {"q": query_str, "year": "t"}
    res = requests.get(f"http://www.wikicfp.com/cfp/servlet/tool.search", params=params)

    soup = BeautifulSoup(res.text, "html.parser")
    for tr in soup.find_all("tr"):
        # direct first child should be a td tag
        td = tr.find("td", recursive=False)
        if not td:
            continue

        # direct first child of td should be a a tag
        a = td.find("a", recursive=False)
        if not a:
            continue

        # href should start with /cfp/servlet/event.showcfp
        if not a["href"].startswith("/cfp/servlet/event.showcfp"):
            continue

        td2 = td.find_next_sibling("td")

        # acronym is the text of the a tag minus the last word (the year)
        acronym = " ".join(a.text.split(" ")[:-1])
        # name is the text of the td2 tag
        name = td2.text.strip()
        # url is the href of the a tag
        url = a["href"]
        
        yield acronym, name, url

def scrape(current: str, output: str, do_series: bool = True, query: Optional[List[List[str]]] = None):
    """Scrape WikiCFP for CFPs

    Args:
        current (str): Path to current csv file
        output (str): Path to output csv file'
        do_series (bool, optional): Whether to get CFPs for all series. Defaults to True.
        query (Optional[List[List[str]]], optional): Query to filter CFPs. Defaults to None.
    """
    if current:
        current = pathlib.Path(current).resolve(strict=True)

    output = pathlib.Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(current)
    skip = set(df["acronym"].tolist())

    rows = [row.tolist() for _, row in df.iterrows()]
    for acronym, name, url, start, end, where, deadline, cfp in get_series_cfps(skip, do_series, query):
        # append to df
        rows.append([acronym, name, url, start, end, where, deadline, cfp])

        df = pd.DataFrame(rows, columns=["acronym", "name", "url", "start", "end", "where", "deadline", "cfp"])

        # set Start, End, and Deadline to datetime - make errors NaT
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")

        # save to csv
        df.to_csv(output, index=False)
     
def load_core_ranks() -> pd.DataFrame:
    """Load the core ranks from the csv file
    
    Returns:
        pd.DataFrame: Core ranks
    """
    df = pd.read_csv(thisdir / "CORE.csv", header=None)
    cols = ["id", "name", "acronym", "source", "rank"]
    # take first len(cols) columns
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    # return without id
    return df.iloc[:, 1:]

def load_CFPs() -> pd.DataFrame:
    # load CFPs
    CFPs = pd.read_csv(thisdir / "CFP.csv")
    # set Start, End, and Deadline to datetime - make errors NaT
    CFPs["start"] = pd.to_datetime(CFPs["start"], errors="coerce")
    CFPs["end"] = pd.to_datetime(CFPs["end"], errors="coerce")
    CFPs["deadline"] = pd.to_datetime(CFPs["deadline"], errors="coerce")

    return CFPs

def augment_CFPs(CFPs: pd.DataFrame, other: pd.DataFrame, on: str, how: str = "left") -> pd.DataFrame:
    """Augment CFPs with other data
    
    Args:
        CFPs (pd.DataFrame): CFPs
        other (pd.DataFrame): Other data
        on (str): Column to join on
        how (str, optional): How to join. Defaults to "left".
    
    Returns:
        pd.DataFrame: Augmented CFPs
    """
    # join on acronym
    CFPs = CFPs.merge(other, on=on, how=how)
    # drop duplicates
    CFPs = CFPs.drop_duplicates()
    # reset index
    CFPs = CFPs.reset_index(drop=True)

    return CFPs

def score(clauses: List[List[str]], cfp: str) -> float:
    return min( # max match across clauses - ALL clauses should match
        max( # max across line and queries - any query can match any line
            fuzz.partial_ratio(
                query.strip().lower(), 
                line.strip().lower()
            )
            for line in cfp.splitlines()
            for query in clause
        )
        for clause in clauses
    )

def load_data() -> pd.DataFrame:
    df = load_CFPs()
    core_ranks = load_core_ranks()
    df = augment_CFPs(df, core_ranks[["acronym", "rank"]], on="acronym")
    df.rename(columns={"rank": "CORE Rank"}, inplace=True)
    df["name"] = df["name"].str.encode("latin-1", errors="ignore").str.decode("utf-8", errors="ignore")
    df["cfp"] = df["cfp"].str.encode("latin-1", errors="ignore").str.decode("utf-8", errors="ignore")
    df["where"] = df["where"].str.encode("latin-1", errors="ignore").str.decode("utf-8", errors="ignore")
    return df

def get_parser() -> argparse.ArgumentParser:
    # create parser
    parser = argparse.ArgumentParser()

    # add scrape subcommand
    subparsers = parser.add_subparsers(dest="subcommand")
    scrape_parser = subparsers.add_parser("scrape")
    scrape_parser.set_defaults(func="scrape")
    scrape_parser.add_argument("--current", help="Path to current CFP.csv file", default=None)
    scrape_parser.add_argument("--output", help="Path to output CFP.csv file", default=thisdir / "CFP.csv")

    # add search subcommand
    search_parser = subparsers.add_parser("search")
    search_parser.set_defaults(func="search")
    search_parser.add_argument("--upcoming", help="Search upcoming conferences", action="store_true")
    search_parser.add_argument("--query", help="Query for a conference", default=None)
    search_parser.add_argument("--query-threshold", help="Threshold for fuzzy matching", default=0.0, type=float)
    search_parser.add_argument(
        "--rank-threshold", help="Threshold for CORE rank", default="N/A", 
        choices=["A*", "A", "B", "C", "N/A"]
    )
    search_parser.add_argument("--print-name", help="Print conference name", action="store_true")

    # add get subcommand
    get_parser = subparsers.add_parser("get")
    get_parser.set_defaults(func="get")
    get_parser.add_argument("acronym", help="Acronym of conference to get")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.func == "scrape":
        scrape(args.current, args.output)
    elif args.func == "search":
        if args.query:
            query = [clause.split(",") for clause in args.query.split(";")]
            scrape(thisdir / "CFP.csv", thisdir / "CFP.csv", do_series=False, query=query)

        df = load_data()

        # sort by CORE Rank
        sortorder = ["A*", "A", "B", "C", "N/A"]
        # if CORE Rank is not in sortorder, make it "N/A"
        df["CORE Rank"] = df["CORE Rank"].apply(lambda rank: rank if rank in sortorder else "N/A")
        df["CORE Rank"] = pd.Categorical(df["CORE Rank"], categories=sortorder, ordered=True)

        if args.rank_threshold != "N/A":
            threshold_int = sortorder.index(args.rank_threshold)
            df = df[df["CORE Rank"].apply(lambda rank: sortorder.index(rank) <= threshold_int)]

        sortby = [("CORE Rank", True)]

        if args.upcoming: # sort by time until conference from now
            df["time_until_conference"] = df["deadline"] - pd.Timestamp.now()
            sortby.insert(0, ("time_until_conference", True))
            # remove negative values
            df = df[df["time_until_conference"] > pd.Timedelta(0)]
        if args.query:
            # split query into clauses. Clauses are separated by ; and each clause is a list of terms separated by ,
            query = [clause.split(",") for clause in args.query.split(";")]
            # score each CFP
            df["query_score"] = df["cfp"].apply(lambda cfp: score(query, cfp))
            # sort by score
            sortby.insert(0, ("query_score", False))

            if args.query_threshold > 0:
                # remove CFPs with score below threshold
                df = df[df["query_score"] >= args.query_threshold]

        # sort by sortby
        df = df.sort_values(by=[col for col, _ in sortby], ascending=[asc for _, asc in sortby])
            
        # only keep relevant columns
        print_cols = ["acronym", "start", "end", "where", "deadline", "CORE Rank", "query_score"]
        if args.print_name:
            print_cols.insert(1, "name")
        df = df[print_cols]
        

        # format dates as Month Day, Year. ignore NaT
        df["start"] = df["start"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)
        df["end"] = df["end"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)
        df["deadline"] = df["deadline"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)
        
        print(df.to_string(index=False))

    elif args.func == "get":
        df = load_data()
        df = df[df["acronym"] == args.acronym]
        
        df["start"] = df["start"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)
        df["end"] = df["end"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)
        df["deadline"] = df["deadline"].apply(lambda date: date.strftime("%b %d, %Y") if not pd.isna(date) else date)

        url = df["url"].values[0]
        url = url.replace(" ", "%20")
        url = f"http://wikicfp.com{url}"
        df["url"] = url

        summary = df[["acronym", "name", "where", "url", "start", "end", "deadline", "CORE Rank"]]

        # rotate summary
        summary = summary.transpose()
        print(summary.to_string(header=False))
        
        # <ACRONYM>: <NAME>
        # <WHERE>
        # <URL>
        # <START> - <END>
        # DEADLINE: <DEADLINE>
        # CORE Rank: <CORE RANK>
        # Call for Papers: 
        #     <CFP>
        print("Call for Papers:")
        print("\n    ".join(df["cfp"].values[0].splitlines()))


if __name__ == "__main__":
    main()