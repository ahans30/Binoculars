import argparse
import json
from pathlib import Path

from .citations import build_title_index, index_library_citations, index_pubmed_citations, reset_citations
from .corpus import build_corpus
from .engine import CheckOptions, check_document
from .integrity import update_integrity
from .library import add_library
from . import DATA_DIR
from .pubmed import DEFAULT_WORKERS, add_pubmed, add_pubmed_query
from .pdfreport import write_pdf
from .pmc import add_pmc_query
from .report import write_html
from .semantic import (
    DEFAULT_THRESHOLD, MAX_PUBMED_DOCS, build_index, build_language_map, build_pubmed_index, extend_index,
    matching_pubmed_docs,
)
from .store import Store
from .triage import triage_report


def main():
    parser = argparse.ArgumentParser(prog="simcheck", description="ローカル類似度チェッカー")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-library", help="手元のPDFフォルダを照合先に登録")
    p.add_argument("folder")
    p.add_argument("--workers", type=int)
    p.add_argument("--rebuild", action="store_true", help="登録済みのPDFを消してから登録し直す")

    p = sub.add_parser("add-pubmed", help="PubMedベースラインを照合先に登録")
    p.add_argument("--first", type=int, help="開始ファイル番号（1始まり）")
    p.add_argument("--last", type=int, help="終了ファイル番号")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--keep-gz", action="store_true")

    p = sub.add_parser("add-pubmed-query", help="検索式に合うPubMed抄録だけを登録（1万件まで）")
    p.add_argument("term")

    p = sub.add_parser("add-pmc-query", help="検索式に合うCCライセンスのPMC全文を登録（1万件まで）")
    p.add_argument("term")

    p = sub.add_parser("build-corpus", help="検索式のファイルからクラウド用の照合先を登録")
    p.add_argument("file")

    sub.add_parser("update-integrity", help="不自然な語句のリストと撤回データを取得")
    sub.add_parser("index-citations", help="引用パターンの確認に使う引用の一覧を作成")
    sub.add_parser("compact", help="指紋の索引を1ファイルに統合")
    p = sub.add_parser("index-semantic", help="言い換え検索用索引を作成（既定は手元のPDF）")
    p.add_argument("--pubmed", action="store_true", help="PubMed抄録の索引を作る")
    p.add_argument("--max-docs", type=int, default=MAX_PUBMED_DOCS, help="PubMed抄録の件数の上限")
    p.add_argument("--languages-only", action="store_true", help="作り直さず、文の言語の情報だけを作る")
    p.add_argument("--add-matching", metavar="正規表現", help="題名か雑誌名が合うPubMed抄録を索引に加える")
    p.add_argument("--add-pmc", action="store_true", help="登録済みのPMC全文を索引に加える")
    sub.add_parser("stats", help="索引の状態を表示")

    p = sub.add_parser("check", help="文書を確認してHTMLレポートを作成")
    p.add_argument("file")
    p.add_argument("--out")
    p.add_argument("--include-references", action="store_true")
    p.add_argument("--include-quotes", action="store_true")
    p.add_argument("--exclude-methods", action="store_true")
    p.add_argument("--exclude-preprints", action="store_true")
    p.add_argument("--exclude-common", action="store_true", help="定型表現を一致に数えない")
    p.add_argument("--semantic-threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--exclude-id", action="append", default=[], help="除外する出典ID（複数指定可）")
    p.add_argument("--triage", action="store_true", help="一致箇所を Gemini で仕分ける（原稿の一致箇所を外部に送信）")

    args = parser.parse_args()
    with Store() as store:
        if args.command == "add-library":
            if args.rebuild:
                store.remove_source("library")
            result = add_library(store, args.folder, workers=args.workers)
            print(f"追加 {result['added']} / 重複 {result['duplicates']} / スキップ {len(result['skipped'])}")
            for path, reason in result["skipped"]:
                print(f"  スキップ: {Path(path).name}: {reason}")
        elif args.command == "add-pubmed":
            result = add_pubmed(store, args.first, args.last, args.workers, args.keep_gz)
            print(f"追加 {result['added']} 件 / 失敗 {len(result['failed'])} ファイル")
        elif args.command == "add-pubmed-query":
            result = add_pubmed_query(store, args.term)
            print(f"検索結果 {result['found']:,} 件 / 追加 {result['added']:,} 件")
        elif args.command == "add-pmc-query":
            result = add_pmc_query(store, args.term)
            print(f"検索結果 {result['found']:,} 件 / 追加 {result['added']:,} 件")
        elif args.command == "build-corpus":
            print(json.dumps(build_corpus(store, args.file), ensure_ascii=False))
        elif args.command == "index-citations":
            print(f"題名 {build_title_index(store):,} 件")
            reset_citations(store)
            print(index_pubmed_citations(store))
            print(index_library_citations(store))
        elif args.command == "update-integrity":
            result = update_integrity()
            print(
                f"語句 {result['tortured']:,} 件（解釈できない式 {result['tortured_skipped']:,} 件）/ "
                f"撤回・懸念表明 {result['retraction_dois']:,} DOI"
            )
        elif args.command == "compact":
            store.compact()
            print(json.dumps(store.stats(), ensure_ascii=False))
        elif args.command == "index-semantic":
            if args.add_matching:
                print(extend_index(store, matching_pubmed_docs(store, args.add_matching)))
            elif args.add_pmc:
                print(extend_index(store, store.iter_doc_ids("pmc")))
            elif args.pubmed:
                print(build_pubmed_index(store, max_docs=args.max_docs))
            elif args.languages_only:
                print(build_language_map(store))
            else:
                print(build_index(store))
        elif args.command == "stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        elif args.command == "check":
            options = CheckOptions(
                exclude_references=not args.include_references,
                exclude_quotes=not args.include_quotes,
                exclude_methods=args.exclude_methods,
                exclude_preprints=args.exclude_preprints,
                exclude_common=args.exclude_common,
                min_words=args.min_words,
                use_semantic=not args.no_semantic,
                semantic_threshold=args.semantic_threshold,
                exclude_ext_ids=set(args.exclude_id),
            )
            report = check_document(args.file, store, options)
            if args.triage:
                report.triage, notices = triage_report(report)
                report.notices += notices
            # 原稿の隣に出すとクラウド同期の対象になりうるので、既定はローカルの data 配下にする
            out = Path(args.out) if args.out else DATA_DIR / "reports" / f"{Path(args.file).stem}_report.html"
            write_html(report, out)
            pdf = write_pdf(report, args.file, out.with_suffix(".pdf"))
            print(f"類似度 {report.score * 100:.1f}% / 出典 {len(report.sources)} 件 / 言い換え候補 {len(report.semantic_hits)} 件")
            print(f"レポート: {pdf}")
            print(f"詳細（出典の本文との比較）: {out}")


if __name__ == "__main__":
    main()
