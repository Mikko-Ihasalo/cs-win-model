import pandas as pd
from demoparser2 import DemoParser
import os
from typing import Tuple

TEAM_NUMBER_SIDE_MAP = {2: "T", 3: "CT"}
BOMB_SITE_MAP = {"A": 1, "B": 2}
COLUMNS_TO_KEEP = [
    "demo_file",
    "tick",
    "round",
    "attacker_X",
    "attacker_Y",
    "attacker_Z",
    "attacker_name",
    "user_X",
    "user_Y",
    "user_Z",
    "user_name",
    "weapon",
    "bomb_site",
    "is_bomb_planted",
    "round_time_left",
    "attacker_alive_count",
    "user_alive_count",
    "attacker_team_name",
    "user_team_name",
    "t_won_round",
]


def initialize_demo_parser(demo_file: str) -> DemoParser:
    """
    Initializes the DemoParser with the given demo file.

    Parameters
    ----------
    demo_file : str
        The path to the demo file.

    Returns
    -------
    DemoParser
        An instance of the DemoParser class.

    """
    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo file {demo_file} not found.")
    parser = DemoParser(demo_file)
    return parser


def parse_round_times(parser: DemoParser) -> pd.DataFrame:
    """
    Parses the demo file to extract round times and ticks for mapping.

    Parameters
    ----------
    parser : DemoParser
        An instance of the DemoParser class.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the round times and ticks.
        Columns: ['round_time', 'tick']

    """
    df_ticks = parser.parse_ticks(
        ["game_time", "is_warmup_period", "round_start_time", "last_place_name"]
    )[
        [
            "tick",
            "game_time",
            "is_warmup_period",
            "round_start_time",
        ]
    ]
    df_ticks = df_ticks[df_ticks["is_warmup_period"] == False]

    df_ticks["round_time"] = df_ticks["game_time"] - df_ticks["round_start_time"]
    df_ticks.drop(
        ["game_time", "is_warmup_period", "round_start_time"], axis=1, inplace=True
    )
    df_ticks = df_ticks.drop_duplicates()
    df_ticks = df_ticks.round({"round_time": 0})

    return df_ticks


def parse_alive_teammates(parser: DemoParser) -> pd.DataFrame:
    """
    Parses the demo file to extract alive teammates and ticks for mapping.

    Parameters
    ----------
    parser : DemoParser
        An instance of the DemoParser class.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the alive teammates and ticks.
        Columns: ["tick", "alive_counts","team_name"]

    """
    df_alive = parser.parse_ticks(["is_alive", "team_num"])
    df_alive.drop(["steamid"], axis=1, inplace=True)
    df_alive = df_alive.groupby(["tick", "team_num"]).agg({"is_alive": ["sum"]})
    df_alive = df_alive.reset_index()
    df_alive.columns = ["tick", "team_num", "alive_count"]
    df_alive["team_name"] = df_alive["team_num"].map(TEAM_NUMBER_SIDE_MAP)
    df_alive.drop(["team_num"], axis=1, inplace=True)

    return df_alive


def ticks_between_rounds(parser: DemoParser) -> Tuple[pd.DataFrame, int]:
    """
    Parses the demo file to extract ticks between rounds.

    Parameters
    ----------
    parser : DemoParser
        An instance of the DemoParser class.

    Returns
    -------
    Tuple[pd.DataFrame,int]
        A tuple containing a DataFrame with ticks between rounds and the first round tick time.

        A DataFrame containing the ticks between rounds.
        Columns: ["reason","round","end_tick","winner","start_tick"]

        The first round tick time.

    """
    df_round_start = parser.parse_events(["round_start"], player=["last_place_name"])[
        0
    ][1]
    first_round_tick_time = (
        df_round_start.loc[df_round_start["round"] == 1]["tick"].astype(int).max()
    )
    df_round_end = parser.parse_events(["round_end"], player=["last_place_name"])[0][1]
    df_round_start = df_round_start.groupby("round").agg({"tick": ["max"]})["tick"]
    df_round_end["round"] = df_round_end["round"].astype(int) - 1
    df_round_end = df_round_end.loc[df_round_end["round"] > 0]
    df_round_end["start_tick"] = df_round_start["max"]
    df_round_end.rename(columns={"tick": "end_tick"}, inplace=True)

    return (df_round_end, first_round_tick_time)


def _map_round(row: pd.Series, df_round: pd.DataFrame) -> pd.Series:
    """
    Helper function to map the round number to the corresponding round in the DataFrame.
    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing the round number.
    df_round : pd.DataFrame
        The DataFrame containing the round information.

    Returns
    -------
    pd.Series
        A Series containing the mapped round information.

    """
    round_row = df_round[
        (df_round["start_tick"] < row["tick"]) & (df_round["end_tick"] >= row["tick"])
    ]
    return round_row["round"].iloc[0] if not round_row.empty else None


def parse_bomb_plant(parser: DemoParser) -> pd.DataFrame:
    """
    Parses the demo file to extract bomb plant events and ticks for mapping.

    Parameters
    ----------
    parser : DemoParser
        An instance of the DemoParser class.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the bomb plant events and ticks.
        Columns: ["tick","bomb_site","round","bomb_plant_time"]
    """
    df_bomb = parser.parse_events(["bomb_planted"], player=["last_place_name"])[0][1]

    df_bomb.drop(["site", "user_name", "user_steamid"], axis=1, inplace=True)
    df_bomb = df_bomb.replace("Bombsite", "", regex=True)
    df_bomb = df_bomb.rename(columns={"user_last_place_name": "bomb_site"})

    df_round_end = ticks_between_rounds(parser)[0]
    df_ticks_ = parse_round_times(parser)
    df_bomb["round"] = df_bomb.apply(lambda row: _map_round(row, df_round_end), axis=1)
    df_bomb = df_bomb.dropna(subset=["round"])
    df_bomb["round"] = df_bomb["round"].astype(int)
    df_bomb = df_bomb.merge(df_ticks_, on="tick", how="left")
    df_bomb["round_time"] = df_bomb["round_time"].astype(int)
    df_bomb.rename(columns={"round_time": "bomb_plant_time"}, inplace=True)

    return df_bomb


def friendly_teammates(parser: DemoParser, gamer_tag: str) -> list:
    """
    Parses the demo file to extract friendly teammates.

    Parameters
    ----------
    parser : DemoParser
        An instance of the DemoParser class.
    gamer_tag : str
        The gamer tag of the player.

    Returns
    -------
    list
        A list containing the friendly teammates.
    """
    friendly_team = parser.parse_player_info()
    friendly_num = friendly_team[friendly_team["name"] == gamer_tag]["team_number"]
    friendly_team = list(
        friendly_team[friendly_team["team_number"] == friendly_num.values[0]]["name"]
    )
    return friendly_team


def parse_demo(demo_file: str) -> pd.DataFrame:
    """
    Parses the demo file to extract relevant information.


    Parameters
    ----------
    demo_file : str
        The path to the demo file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parsed information.
    """
    parser = initialize_demo_parser(demo_file)
    df_ticks = parse_round_times(parser)
    df_alive = parse_alive_teammates(parser)
    df_bomb = parse_bomb_plant(parser)
    df_round_end, first_round_tick_time = ticks_between_rounds(parser)
    df = parser.parse_event(
        "player_death",
        player=["X", "Y", "Z", "pending_team_num"],
        other=["game_time", "team_name"],
    )
    df = df.dropna(subset=["attacker_name"])

    df = df.merge(df_ticks, on="tick", how="left")

    df_bomb = df_bomb.merge(df_ticks, on="tick", how="left")

    df = df.loc[df["tick"] > first_round_tick_time]
    df = df.loc[df["tick"] <= df_round_end["end_tick"].astype(int).max()]

    df["round"] = df.apply(lambda row: _map_round(row, df_round_end), axis=1)
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)
    df = df.merge(df_round_end[["reason", "round", "winner"]], on=["round"], how="left")

    df = df.merge(df_bomb, on=["round"], how="left", suffixes=("_kill", "_bomb"))
    df["is_bomb_planted"] = (df["tick_bomb"] <= df["tick_kill"]).astype(int)

    df["round_time_left"] = df.apply(
        lambda row: (
            115 - row["round_time_kill"]
            if not row["is_bomb_planted"]
            else 40 - (row["round_time_kill"] - row["bomb_plant_time"])
        ),
        axis=1,
    )

    df["attacker_team_name"] = df["attacker_pending_team_num"].map(TEAM_NUMBER_SIDE_MAP)
    df["user_team_name"] = df["user_pending_team_num"].map(TEAM_NUMBER_SIDE_MAP)
    df.drop(
        ["attacker_pending_team_num", "user_pending_team_num"], axis=1, inplace=True
    )

    df = df.merge(
        df_alive,
        left_on=["tick_kill", "attacker_team_name"],
        right_on=["tick", "team_name"],
        how="left",
    )
    df.rename(columns={"alive_count": "attacker_alive_count"}, inplace=True)
    df["attacker_alive_count"] = (
        df["attacker_alive_count"] - 1
    )  # We only consider alive teammates.

    df.drop(["tick", "team_name"], axis=1, inplace=True)
    df = df.merge(
        df_alive,
        left_on=["tick_kill", "user_team_name"],
        right_on=["tick", "team_name"],
        how="left",
    )
    df.rename(columns={"alive_count": "user_alive_count"}, inplace=True)
    # df.drop(["tick", "team_name"], axis=1, inplace=True)

    df["is_t_attacker"] = (df["attacker_team_name"] == "T").astype(int)

    df["t_won_round"] = ("T" == df["winner"]).astype(int)
    df.drop(
        ["tick_kill", "winner", "tick_bomb", "reason", "bomb_plant_time"],
        axis=1,
        inplace=True,
    )
    df["demo_file"] = demo_file.split("\\")[-1]
    df["bomb_site"] = df["bomb_site"].map(BOMB_SITE_MAP).fillna(0).astype(int)

    df = df[COLUMNS_TO_KEEP]
    df = df.drop_duplicates()

    return df


def parse_multiple_demos(demo_folder_path: str, map=str) -> pd.DataFrame:
    """
    Parses multiple demo files in the specified folder and combines the results into a single DataFrame.

    Parameters
    ----------
    demo_folder_path : str
        The path to the folder containing the demo files.
    gamer_tag : str
        The gamer tag of the player.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parsed information from all demo files.
    """
    all_demos = []
    for i, demo_file in enumerate(os.listdir(demo_folder_path)):
        if demo_file.endswith(".dem"):
            demo_path = os.path.join(demo_folder_path, demo_file)
            parser = initialize_demo_parser(demo_path)
            server_info = parser.parse_header()
            if server_info["map_name"] != map:
                print(f"Skipping demo {i+1} : {demo_file} (map mismatch)")
                continue
            # if "FACEIT" not in server_info["server_name"]:
            #    print(f"Skipping demo {i+1} : {demo_file} (not a FACEIT demo)")
            #    continue
            print(f"Parsing demo {i+1} : {demo_path}")
            demo_data = parse_demo(demo_path)
            print(f"Demo  {i+1}  parsing complete.")
            all_demos.append(demo_data)

    print(f"Parsing complete. {len(all_demos)} demos parsed.")
    return pd.concat(all_demos, ignore_index=True)


def save_parsed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the parsed DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the parsed information.
    output_path : str
        The path to save the CSV file.

    Returns
    -------
    None
    """
    df.to_csv(output_path, index=False)
    print(f"Parsed data saved to {output_path}")
