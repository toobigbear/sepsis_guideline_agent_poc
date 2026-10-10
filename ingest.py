"""知识库入库：解析 data/kb/*.md -> 层次化语义存储（SQLite）。

用法：python ingest.py
"""
import glob
import os
import sys

import config
from vikingrag.storage import Storage, parse_markdown


def main() -> None:
    if os.path.exists(config.DB_PATH):
        os.remove(config.DB_PATH)
    st = Storage(config.DB_PATH)
    files = sorted(glob.glob(os.path.join(config.KB_DIR, "*.md")))
    if not files:
        sys.exit("data/kb 下没有 Markdown 文档")
    for path in files:
        tree = parse_markdown(path)
        stats = st.add_document(tree)
        print(f"已入库 {os.path.basename(path):32s} "
              f"节点 {stats['nodes']:3d} 个，对象 {stats['objects']:3d} 个")
    print(f"\n存储概览：")
    print(f"  DeepRead 式基线（完整目录入 prompt）：约 {st.full_catalog_tokens()} token")
    print(f"  数据库文件：{config.DB_PATH}")
    st.close()


if __name__ == "__main__":
    main()
