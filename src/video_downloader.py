#!/usr/bin/env python3
"""
Source code from https://github.com/visurg-ai/LEMON
Modified it to read SurgLaVi SQL database
"""

import argparse
import os
import sqlite3
import multiprocessing

# Third-party / project deps (unchanged from your original flow)
import you2dl


def help_text(short_option):
    return {
        '-c': 'Path to a cookies file for authenticated downloads',
        '-h': 'Display help',
        '-n': 'Max number of videos to download (optional)',
        '-o': 'Path to the output folder (REQUIRED)',
        '-e': 'Directory containing previously downloaded videos to avoid re-downloading',
        '-s': 'Skip video download (e.g., only fetch metadata if supported)',
        '-d': 'Additionally download/keep description',
    }[short_option]


def parse_cmdline_params():
    parser = argparse.ArgumentParser(description='Download YouTube videos listed in an SQLite database.')
    parser.add_argument('--db', required=True, type=str,
                        help='Path to the SQLite database file (e.g., ../data/surglavi_beta.db)')
    parser.add_argument('--table', default='videos', type=str,
                        help='Table name containing the youtube_id column (default: videos)')
    parser.add_argument('--id-col', default='youtube_id', type=str,
                        help='Column name that stores the YouTube video ID or URL (default: youtube_id)')
    parser.add_argument('--name-col', default='name', type=str,
                        help='Optional: video name column (used only for logging; default: name)')
    parser.add_argument('-n', '--number', default=None, type=int, help=help_text('-n'))
    parser.add_argument('-o', '--output', required=True, type=str, help=help_text('-o'))
    parser.add_argument('-e', '--existing', default=None, type=str, help=help_text('-e'))
    parser.add_argument('-c', '--cookies', default=None, type=str, help=help_text('-c'))
    parser.add_argument('-s', '--skip-video', action='store_true', help=help_text('-s'))
    parser.add_argument('-d', '--description', action='store_true', help=help_text('-d'))
    parser.add_argument('--without-audio', action='store_true')
    parser.add_argument('--audio-separated', action='store_true')
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--allow-existing-output', action='store_true',
                        help='If set, do not error when output folder already exists.')
    args = parser.parse_args()

    # Minimal validation
    if not args.output:
        raise ValueError("[ERROR] An output folder has not been provided.")
    return args


def normalize_to_url(yid: str) -> str:
    """
    Accepts either a bare 11-char YouTube ID or a full/short YouTube URL.
    Returns a standard watch URL.
    """
    if not yid:
        return None
    s = yid.strip()
    if s.startswith('http://') or s.startswith('https://'):
        return s
    # fallback: treat as ID
    return f"https://www.youtube.com/watch?v={s}"


def read_youtube_ids_from_db(db_path: str, table: str, id_col: str, name_col: str, limit: int | None):
    """
    Read youtube IDs (or URLs) from the given SQLite database.
    Uses read-only URI and ignores NULL/empty IDs. Returns list of URLs.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    cur = conn.cursor()

    # Build and execute query
    q = f"""
        SELECT {id_col}, {name_col}
        FROM {table}
        WHERE {id_col} IS NOT NULL AND TRIM({id_col}) != ''
    """
    if limit is not None and limit > 0:
        q += f" LIMIT {limit}"

    cur.execute(q)
    rows = cur.fetchall()
    conn.close()

    urls = []
    for yid, name in rows:
        url = normalize_to_url(yid)
        if url:
            urls.append(url)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def main():
    # Read command line parameters
    args = parse_cmdline_params()

    # Create destination folder
    try:
        os.mkdir(args.output)
    except FileExistsError:
        if not args.allow_existing_output:
            raise ValueError('[ERROR] Output directory already exists. Use --allow-existing-output to continue.')

    # Collect list of already downloaded videos
    already_downloaded = []
    if args.existing is not None:
        # Reuse your helper to detect previously downloaded list from JSON manifests
        already_downloaded = you2dl.find_all_videos_byjson(args.existing)

    # Build video URL list from database
    video_urls = read_youtube_ids_from_db(
        db_path=args.db,
        table=args.table,
        id_col=args.id_col,
        name_col=args.name_col,
        limit=args.number
    )

    # Prune against already downloaded
    if already_downloaded:
        video_urls = you2dl.prune_video_list(video_urls, already_downloaded)

    print('[INFO] Videos to be downloaded:', len(video_urls))

    if not video_urls:
        print('[INFO] Nothing to download. Exiting.')
        return

    # Download with a process pool (same call signature as your original script)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = [
        pool.apply_async(
            you2dl.download,
            args=(
                url,
                args.output,
                args.cookies,
                3,  # retries
                args.skip_video,
                not args.description,  # your original used "not args.description"
                args.without_audio,
                args.audio_separated,
                args.audio_only
            )
        )
        for url in video_urls
    ]
    pool.close()
    pool.join()

    # Optionally surface any errors from async calls
    failures = []
    for r in results:
        try:
            r.get()
        except Exception as ex:
            failures.append(str(ex))
    if failures:
        print(f'[WARN] {len(failures)} downloads reported errors:')
        for f in failures[:10]:
            print('   -', f)
        if len(failures) > 10:
            print('   ...')


if __name__ == '__main__':
    main()
