import os
import time
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import isodate
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "crawler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Virality Features Collected ──────────────────────────────────────────────
# 
# ENGAGEMENT
#   view_count, like_count, comment_count
#   like_rate (likes/views), comment_rate (comments/views)
#   engagement_rate (likes+comments)/views
#
# CONTENT
#   title_length, title_has_number, title_has_question, title_has_caps_word
#   description_length, tag_count
#   duration_seconds, category_id, category_name, default_language
#   has_captions, definition (hd/sd), caption_count
#
# CHANNEL
#   channel_subscriber_count, channel_video_count, channel_view_count
#   channel_age_days
#
# TIMING
#   published_at, publish_hour, publish_day_of_week, publish_month
#
# THUMBNAIL
#   thumbnail_url (for manual or ML analysis)
#
# VELOCITY (computed at crawl time)
#   days_since_publish, views_per_day
# ─────────────────────────────────────────────────────────────────────────────


CATEGORY_MAP = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
    "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
    "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism",
}


def build_youtube(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def parse_duration(iso_duration: str) -> int:
    """Convert ISO 8601 duration (PT4M13S) → seconds."""
    try:
        return int(isodate.parse_duration(iso_duration).total_seconds())
    except Exception:
        return 0


def extract_title_features(title: str) -> dict:
    words = title.split()
    return {
        "title_length": len(title),
        "title_word_count": len(words),
        "title_has_number": any(ch.isdigit() for ch in title),
        "title_has_question": "?" in title,
        "title_has_exclamation": "!" in title,
        "title_all_caps_word": any(w.isupper() and len(w) > 2 for w in words),
        "title_has_brackets": ("[" in title or "(" in title),
        "title_emoji_count": sum(1 for c in title if ord(c) > 127462),
    }


def extract_video_id(item: dict) -> str:
    """Safely extract video ID whether item['id'] is a string or a dict."""
    raw = item.get("id", "")
    if isinstance(raw, dict):
        return raw.get("videoId", "")
    return str(raw)


def is_valid_video_id(vid_id: str) -> bool:
    """YouTube video IDs are exactly 11 characters, alphanumeric + - _"""
    import re
    return bool(vid_id and re.match(r'^[A-Za-z0-9_\-]{11}$', vid_id))


def enrich_video(item: dict, channel_stats: dict, now: datetime, region: str = "") -> dict:
    """Flatten a YouTube API video item into a feature row."""
    vid_id = extract_video_id(item)
    if not is_valid_video_id(vid_id):
        raise ValueError(f"Invalid video ID: {repr(vid_id)}")
    snip = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})
    status = item.get("status", {})

    views    = safe_int(stats.get("viewCount"))
    likes    = safe_int(stats.get("likeCount"))
    comments = safe_int(stats.get("commentCount"))

    published_raw = snip.get("publishedAt", "")
    try:
        pub_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
    except Exception:
        pub_dt = now

    days_live = max((now - pub_dt).days, 1)

    title = snip.get("title", "")
    description = snip.get("description", "")
    tags = snip.get("tags", [])
    cat_id = snip.get("categoryId", "")

    chan = channel_stats.get(snip.get("channelId", ""), {})

    row = {
        # Crawl metadata
        "crawl_date":             now.strftime("%Y-%m-%d"),
        "region":                 region.upper(),
        "days_trending":          1,

        # Identity
        "video_id":               vid_id,
        "title":                  title,
        "channel_id":             snip.get("channelId"),
        "channel_title":          snip.get("channelTitle"),

        # Timing
        "published_at":           published_raw,
        "publish_hour":           pub_dt.hour,
        "publish_day_of_week":    pub_dt.strftime("%A"),
        "publish_month":          pub_dt.month,
        "days_since_publish":     days_live,

        # Engagement (raw)
        "view_count":             views,
        "like_count":             likes,
        "comment_count":          comments,

        # Engagement (rates)
        "like_rate":              round(likes / views, 6) if views else 0,
        "comment_rate":           round(comments / views, 6) if views else 0,
        "engagement_rate":        round((likes + comments) / views, 6) if views else 0,

        # Velocity
        "views_per_day":          round(views / days_live, 2),

        # Content metadata
        "description_length":     len(description),
        "tag_count":              len(tags),
        "tags":                   "|".join(tags[:30]),   # top 30 tags
        "category_id":            cat_id,
        "category_name":          CATEGORY_MAP.get(cat_id, "Unknown"),
        "duration_seconds":       parse_duration(content.get("duration", "")),
        "definition":             content.get("definition", ""),      # hd / sd
        "has_captions":           content.get("caption", "false") == "true",
        "default_language":       snip.get("defaultAudioLanguage", snip.get("defaultLanguage", "")),
        "license":                status.get("license", ""),
        "made_for_kids":          status.get("madeForKids", False),

        # Thumbnail
        "thumbnail_url":          (snip.get("thumbnails", {}).get("maxres")
                                   or snip.get("thumbnails", {}).get("high")
                                   or {}).get("url", ""),

        # Channel signals
        "channel_subscriber_count": safe_int(chan.get("subscriberCount")),
        "channel_video_count":      safe_int(chan.get("videoCount")),
        "channel_view_count":       safe_int(chan.get("viewCount")),
        "channel_age_days":         chan.get("channel_age_days", 0),

        # URL
        "url": f"https://www.youtube.com/watch?v={vid_id}",
    }

    row.update(extract_title_features(title))
    return row


# ─── API Helpers ──────────────────────────────────────────────────────────────

def fetch_channel_stats(youtube, channel_ids: list[str]) -> dict:
    """Batch fetch channel statistics (max 50 per call)."""
    result = {}
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        try:
            resp = youtube.channels().list(
                part="statistics,snippet",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            log.warning(f"Channel fetch error: {e}")
            continue

        for item in resp.get("items", []):
            cid = item["id"]
            stats = item.get("statistics", {})
            snip = item.get("snippet", {})
            created = snip.get("publishedAt", "")
            try:
                age = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(created.replace("Z", "+00:00"))).days
            except Exception:
                age = 0
            result[cid] = {**stats, "channel_age_days": age}
    return result


def fetch_video_details(youtube, video_ids: list[str]) -> list[dict]:
    """Fetch full details for up to 50 video IDs."""
    import re
    # Validate IDs before sending — a single bad ID can silently break the whole batch
    clean_ids = [vid for vid in video_ids if re.match(r'^[A-Za-z0-9_\-]{11}$', str(vid))]
    skipped = len(video_ids) - len(clean_ids)
    if skipped:
        log.warning(f"Skipped {skipped} malformed video IDs before API call.")
    items = []
    for i in range(0, len(clean_ids), 50):
        batch = clean_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="snippet,statistics,contentDetails,status",
                id=",".join(batch),
            ).execute()
            items.extend(resp.get("items", []))
        except HttpError as e:
            log.warning(f"Video details error: {e}")
    return items


def search_videos(youtube, query: str, max_results: int = 200,
                  order: str = "relevance", region: str = "US") -> list[str]:
    """Return video IDs matching a search query."""
    ids, page_token = [], None
    pbar = tqdm(total=max_results, desc=f"Search: {query}")

    while len(ids) < max_results:
        fetch_n = min(50, max_results - len(ids))
        try:
            resp = youtube.search().list(
                part="id",
                q=query,
                type="video",
                order=order,
                regionCode=region,
                maxResults=fetch_n,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            log.error(f"Search error: {e}")
            break

        batch = []
        for item in resp.get("items", []):
            item_id = item.get("id", {})
            # item_id can be a dict (search) or plain string (videos.list)
            if isinstance(item_id, dict):
                if item_id.get("kind") == "youtube#video":
                    vid = item_id.get("videoId", "")
                    if vid:
                        batch.append(vid)
            elif isinstance(item_id, str) and item_id:
                batch.append(item_id)
        ids.extend(batch)
        pbar.update(len(batch))

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.5)

    pbar.close()
    return ids[:max_results]


def fetch_trending(youtube, max_results: int = 200, region: str = "US",
                   category_id: str = "0") -> list[str]:
    """Return trending video IDs for a region."""
    ids, page_token = [], None
    pbar = tqdm(total=max_results, desc=f"Trending [{region}]")

    while len(ids) < max_results:
        fetch_n = min(50, max_results - len(ids))
        try:
            resp = youtube.videos().list(
                part="id",
                chart="mostPopular",
                regionCode=region,
                videoCategoryId=category_id,
                maxResults=fetch_n,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            log.error(f"Trending error: {e}")
            break

        batch = []
        for item in resp.get("items", []):
            vid = item.get("id", "")
            if isinstance(vid, dict):
                vid = vid.get("videoId", "")
            if vid and isinstance(vid, str):
                batch.append(vid)
        ids.extend(batch)
        pbar.update(len(batch))

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)

    pbar.close()
    return ids[:max_results]


def fetch_channel_videos(youtube, channel_id: str, max_results: int = 200) -> list[str]:
    """Return recent video IDs from a channel."""
    # First, get the uploads playlist ID
    try:
        resp = youtube.channels().list(
            part="contentDetails", id=channel_id
        ).execute()
        uploads_playlist = (resp["items"][0]["contentDetails"]
                            ["relatedPlaylists"]["uploads"])
    except (HttpError, KeyError, IndexError) as e:
        log.error(f"Could not get uploads playlist: {e}")
        return []

    ids, page_token = [], None
    pbar = tqdm(total=max_results, desc=f"Channel: {channel_id}")

    while len(ids) < max_results:
        fetch_n = min(50, max_results - len(ids))
        try:
            resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=fetch_n,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            log.error(f"Playlist error: {e}")
            break

        batch = [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
        ids.extend(batch)
        pbar.update(len(batch))

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)

    pbar.close()
    return ids[:max_results]


# ─── Core Crawler ─────────────────────────────────────────────────────────────

class YouTubeCrawler:
    def __init__(self, api_key: str):
        self.youtube = build_youtube(api_key)
        self.now = datetime.now(timezone.utc)
        # Track (video_id, region) so same video can trend in multiple countries
        self._seen: set[tuple[str, str]] = set()
        self._load_seen()

    def _load_seen(self):
        """Load existing (video_id, region) pairs from CSV."""
        existing = OUTPUT_DIR / "videos.csv"
        if existing.exists():
            try:
                df = pd.read_csv(existing, usecols=["video_id", "region"])
                self._seen = set(
                    zip(df["video_id"].astype(str), df["region"].astype(str).str.upper())
                )
                log.info(f"Loaded {len(self._seen)} existing (video, region) pairs.")
            except Exception:
                pass

    def crawl(self, video_ids: list[str], region: str = "") -> pd.DataFrame:
        region = region.upper()
        new_ids, repeat_ids = [], []

        for vid in video_ids:
            key = (vid, region)
            if key in self._seen:
                repeat_ids.append(vid)
            else:
                new_ids.append(vid)

        log.info(
            f"[{region}] {len(new_ids)} new videos | "
            f"{len(repeat_ids)} still trending (days_trending +1)"
        )

        # ── Step 1: Increment days_trending for repeated videos ──────────────
        if repeat_ids:
            self._increment_days_trending(repeat_ids, region)

        # ── Step 2: Fetch and save brand-new videos ───────────────────────────
        if not new_ids:
            return pd.DataFrame()

        rows = []
        for i in tqdm(range(0, len(new_ids), 50), desc=f"Fetching [{region}]"):
            batch_ids = new_ids[i:i+50]
            items = fetch_video_details(self.youtube, batch_ids)

            channel_ids = list({it["snippet"]["channelId"] for it in items})
            chan_stats = fetch_channel_stats(self.youtube, channel_ids)

            for item in items:
                try:
                    row = enrich_video(item, chan_stats, self.now, region)
                    rows.append(row)
                    self._seen.add((row["video_id"], region))
                except Exception as e:
                    log.warning(f"Skipping {item.get('id')}: {e}")
            time.sleep(0.3)

        return pd.DataFrame(rows)

    def _increment_days_trending(self, video_ids: list[str], region: str):
        """For videos already in the CSV, increment their days_trending by 1."""
        csv_path = OUTPUT_DIR / "videos.csv"
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            mask = (
                df["video_id"].astype(str).isin(video_ids) &
                (df["region"].astype(str).str.upper() == region)
            )
            count = mask.sum()
            if count:
                df.loc[mask, "days_trending"] = df.loc[mask, "days_trending"].fillna(1) + 1
                df.to_csv(csv_path, index=False)
                log.info(f"  ↑ Incremented days_trending for {count} videos in {region}.")
        except Exception as e:
            log.warning(f"Could not update days_trending: {e}")

    def save(self, df: pd.DataFrame, label: str = "videos"):
        if df.empty:
            log.info("Nothing new to save.")
            return

        csv_path = OUTPUT_DIR / "videos.csv"
        json_path = OUTPUT_DIR / "videos.json"

        # Append new rows to CSV
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # Append to JSON (line-delimited)
        with open(json_path, "a") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec) + "\n")

        log.info(f"Saved {len(df)} new rows → {csv_path}")
        self._print_summary(df)

    def _print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print(f"  Crawled {len(df)} videos")
        print(f"  Avg views:          {df['view_count'].mean():,.0f}")
        print(f"  Avg engagement:     {df['engagement_rate'].mean():.4f}")
        print(f"  Avg views/day:      {df['views_per_day'].mean():,.1f}")
        print(f"  Top category:       {df['category_name'].mode().iloc[0]}")
        print(f"  % HD:               {(df['definition']=='hd').mean()*100:.1f}%")
        print(f"  % Has captions:     {df['has_captions'].mean()*100:.1f}%")
        print("=" * 60 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YouTube Virality Crawler")
    parser.add_argument("--mode", choices=["search", "trending", "channel"],
                        required=True, help="Crawl mode")
    parser.add_argument("--query", type=str, default="",
                        help="Search query (for --mode search)")
    parser.add_argument("--channel-id", type=str, default="",
                        help="Channel ID (for --mode channel)")
    parser.add_argument("--max", type=int, default=200,
                        help="Max videos to collect")
    parser.add_argument("--region", type=str, default="US",
                        help="Region code (e.g. US, GB, JP)")
    parser.add_argument("--category", type=str, default="0",
                        help="Category ID filter (trending mode, 0=all)")
    parser.add_argument("--order", type=str, default="relevance",
                        choices=["relevance", "viewCount", "date", "rating"],
                        help="Search result ordering")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not set. Add it to your .env file.")
        return

    youtube = build_youtube(API_KEY)
    crawler = YouTubeCrawler(API_KEY)

    if args.mode == "search":
        if not args.query:
            print("ERROR: --query required for search mode.")
            return
        ids = search_videos(youtube, args.query, args.max, args.order, args.region)

    elif args.mode == "trending":
        ids = fetch_trending(youtube, args.max, args.region, args.category)

    elif args.mode == "channel":
        if not args.channel_id:
            print("ERROR: --channel-id required for channel mode.")
            return
        ids = fetch_channel_videos(youtube, args.channel_id, args.max)

    df = crawler.crawl(ids, region=args.region)
    crawler.save(df, label=args.mode)


if __name__ == "__main__":
    main()