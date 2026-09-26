#!/usr/bin/env python3
"""PNG チャンク構造をフルデコードせずに比較する。

IDAT は連結して zlib ヘッダを読み、先頭 5 行分だけ増分 inflate して打ち切る。
"""
from __future__ import annotations

import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"

COLOR_TYPE_SAMPLES = {
    0: 1,  # grayscale
    2: 3,  # RGB
    3: 1,  # palette
    4: 2,  # gray + alpha
    6: 4,  # RGBA
}

COLOR_TYPE_NAME = {
    0: "grayscale",
    2: "truecolor RGB",
    3: "indexed",
    4: "grayscale+alpha",
    6: "truecolor RGBA",
}

ANCILLARY_KEYS = ("pHYs", "gAMA", "sRGB", "iCCP", "tEXt", "tIME")

FILTER_NAME = {
    0: "None",
    1: "Sub",
    2: "Up",
    3: "Average",
    4: "Paeth",
}


def samples_per_pixel(color_type: int) -> int:
    if color_type not in COLOR_TYPE_SAMPLES:
        raise ValueError(f"unknown color_type={color_type}")
    return COLOR_TYPE_SAMPLES[color_type]


def row_bytes(width: int, bit_depth: int, color_type: int) -> int:
    """フィルタバイト込みの非圧縮 1 行サイズ。"""
    spp = samples_per_pixel(color_type)
    bits = width * spp * bit_depth
    return 1 + (bits + 7) // 8


def parse_ihdr(data: bytes) -> dict:
    if len(data) != 13:
        raise ValueError(f"IHDR length {len(data)} != 13")
    w, h, bit_depth, color_type, compression, filt, interlace = struct.unpack(
        ">IIBBBBB", data
    )
    return {
        "width": w,
        "height": h,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "color_type_name": COLOR_TYPE_NAME.get(color_type, "?"),
        "compression": compression,
        "filter": filt,
        "interlace": interlace,
    }


def parse_phys(data: bytes) -> dict | None:
    if len(data) < 9:
        return None
    x, y, unit = struct.unpack(">IIB", data[:9])
    return {"ppu_x": x, "ppu_y": y, "unit": unit}


def zlib_header_info(cmf: int, flg: int) -> dict:
    cm = cmf & 0x0F
    cinfo = (cmf >> 4) & 0x0F
    fcheck_ok = ((cmf << 8) + flg) % 31 == 0
    fdict = (flg >> 5) & 1
    flevel = (flg >> 6) & 3
    window_bits = cinfo + 8
    window_size = 1 << window_bits
    return {
        "cmf": cmf,
        "flg": flg,
        "cm": cm,
        "cinfo": cinfo,
        "window_bits": window_bits,
        "window_size": window_size,
        "fcheck_ok": fcheck_ok,
        "fdict": fdict,
        "flevel": flevel,
        "flevel_name": {0: "fastest", 1: "fast", 2: "default", 3: "max"}.get(
            flevel, "?"
        ),
    }


def read_chunks(path: Path) -> tuple[list[tuple[str, int, int]], int]:
    """(type, size, file_offset_of_data) のリストとファイルサイズ。"""
    file_size = path.stat().st_size
    chunks: list[tuple[str, int, int]] = []
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != PNG_SIG:
            raise ValueError(f"not a PNG: {path}")
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length, tag = struct.unpack(">I4s", header)
            try:
                name = tag.decode("ascii")
            except UnicodeDecodeError:
                name = tag.hex()
            data_off = f.tell()
            f.seek(length + 4, 1)  # data + CRC
            chunks.append((name, length, data_off))
            if name == "IEND":
                break
    return chunks, file_size


def summarize_chunk_list(chunks: list[tuple[str, int, int]]) -> list[str]:
    lines: list[str] = []
    i = 0
    n = len(chunks)
    while i < n:
        name, size, _ = chunks[i]
        if name != "IDAT":
            lines.append(f"  {name:8s} {size:>12d}")
            i += 1
            continue
        sizes: list[int] = []
        while i < n and chunks[i][0] == "IDAT":
            sizes.append(chunks[i][1])
            i += 1
        total = sum(sizes)
        mean = total / len(sizes)
        lines.append(
            f"  IDAT     {len(sizes):>6d} chunks  "
            f"min={min(sizes)}  max={max(sizes)}  mean={mean:.1f}  "
            f"total_compressed={total}"
        )
    return lines


def inflate_first_rows(
    path: Path,
    idat_chunks: list[tuple[str, int, int]],
    nbytes_needed: int,
) -> tuple[bytes, dict]:
    """IDAT を連結せずチャンク単位で inflate。先頭 nbytes で打ち切る。"""
    if not idat_chunks:
        return b"", {"error": "no IDAT"}

    header_info: dict | None = None
    dec = zlib.decompressobj()
    out = bytearray()
    fed = 0
    chunks_used = 0

    with path.open("rb") as f:
        for name, size, off in idat_chunks:
            f.seek(off)
            piece = f.read(size)
            if header_info is None and len(piece) >= 2:
                header_info = zlib_header_info(piece[0], piece[1])
            fed += len(piece)
            chunks_used += 1
            try:
                out.extend(dec.decompress(piece, max_length=nbytes_needed - len(out)))
            except zlib.error as exc:
                return bytes(out), {
                    "error": f"inflate failed after {fed} compressed bytes: {exc}",
                    "header": header_info,
                    "inflated_bytes": len(out),
                    "compressed_fed": fed,
                    "chunks_used": chunks_used,
                }
            if len(out) >= nbytes_needed:
                break
            if dec.eof:
                break

    info = {
        "header": header_info,
        "inflated_bytes": len(out),
        "needed_bytes": nbytes_needed,
        "enough_for_rows": len(out) >= nbytes_needed,
        "compressed_fed": fed,
        "chunks_used": chunks_used,
        "unused_in_last_chunk": len(dec.unconsumed_tail) if hasattr(dec, "unconsumed_tail") else None,
    }
    return bytes(out[:nbytes_needed]), info


def analyze(path: Path) -> dict:
    chunks, file_size = read_chunks(path)
    types = [c[0] for c in chunks]
    sizes = [c[1] for c in chunks]
    type_counts = Counter(types)
    idat = [c for c in chunks if c[0] == "IDAT"]
    idat_sizes = [c[1] for c in idat]

    ihdr_chunk = next((c for c in chunks if c[0] == "IHDR"), None)
    if ihdr_chunk is None:
        raise ValueError(f"no IHDR in {path}")
    with path.open("rb") as f:
        f.seek(ihdr_chunk[2])
        ihdr = parse_ihdr(f.read(ihdr_chunk[1]))

        extra: dict[str, object] = {}
        for key in ANCILLARY_KEYS:
            found = next((c for c in chunks if c[0] == key), None)
            extra[key] = None
            if found is None:
                continue
            f.seek(found[2])
            raw = f.read(found[1])
            extra[key] = {"size": found[1], "raw_preview": raw[:64]}
            if key == "pHYs":
                extra[key]["parsed"] = parse_phys(raw)
            elif key == "gAMA" and len(raw) == 4:
                extra[key]["gamma"] = struct.unpack(">I", raw)[0] / 100000.0
            elif key == "sRGB" and len(raw) >= 1:
                extra[key]["rendering_intent"] = raw[0]
            elif key == "tEXt":
                extra[key]["text"] = raw.decode("latin-1", errors="replace")[:200]
            elif key == "tIME" and len(raw) == 7:
                y, mo, d, h, mi, s = struct.unpack(">HBBBBB", raw)
                extra[key]["timestamp"] = f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}"

    rb = row_bytes(ihdr["width"], ihdr["bit_depth"], ihdr["color_type"])
    need = rb * 5
    inflated, zinfo = inflate_first_rows(path, idat, need)

    filters: list[int] = []
    for i in range(5):
        off = i * rb
        if off < len(inflated):
            filters.append(inflated[off])
        else:
            break

    return {
        "path": str(path),
        "exists": True,
        "file_size": file_size,
        "ihdr": ihdr,
        "row_bytes": rb,
        "pixel_bytes": rb - 1,
        "chunks": chunks,
        "chunk_types": types,
        "chunk_count": len(chunks),
        "idat_count": len(idat),
        "idat_sizes": idat_sizes,
        "idat_total": sum(idat_sizes),
        "idat_min": min(idat_sizes) if idat_sizes else None,
        "idat_max": max(idat_sizes) if idat_sizes else None,
        "idat_mean": (sum(idat_sizes) / len(idat_sizes)) if idat_sizes else None,
        "first8": types[:8],
        "last4": types[-4:],
        "ancillary": extra,
        "zlib": zinfo,
        "filters": filters,
        "type_counts": dict(type_counts),
        "all_chunk_sizes": sizes,
    }


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n} B ({n / 1024:.1f} KiB)"
    if n < 1024 ** 3:
        return f"{n} B ({n / (1024 ** 2):.2f} MiB)"
    return f"{n} B ({n / (1024 ** 3):.2f} GiB)"


def print_report(label: str, info: dict | None, missing_path: Path | None = None) -> None:
    print("=" * 78)
    if info is None:
        print(f"{label}: MISSING  {missing_path}")
        return
    ih = info["ihdr"]
    print(f"{label}: {info['path']}")
    print(f"  file size          : {fmt_bytes(info['file_size'])}")
    print(
        f"  IHDR               : {ih['width']} x {ih['height']}  "
        f"bit_depth={ih['bit_depth']}  color_type={ih['color_type']} "
        f"({ih['color_type_name']})"
    )
    print(
        f"                     : compression={ih['compression']}  "
        f"filter={ih['filter']}  interlace={ih['interlace']}"
    )
    print(
        f"  uncompressed row   : {info['row_bytes']} bytes "
        f"(1 filter + {info['pixel_bytes']} pixel bytes)"
    )
    print(f"  estimated IDAT #   : {info['idat_count']}")
    print(f"  chunk sequence:")
    for line in summarize_chunk_list(info["chunks"]):
        print(line)
    print(f"  first 8 chunks     : {info['first8']}")
    print(f"  last 4 chunks      : {info['last4']}")
    print("  ancillary present:")
    for key in ANCILLARY_KEYS:
        val = info["ancillary"].get(key)
        if val is None:
            print(f"    {key:6s}: no")
            continue
        extra = ""
        if key == "pHYs" and val.get("parsed"):
            p = val["parsed"]
            extra = f"  ppu=({p['ppu_x']},{p['ppu_y']}) unit={p['unit']}"
        elif key == "gAMA" and "gamma" in val:
            extra = f"  gamma={val['gamma']}"
        elif key == "sRGB" and "rendering_intent" in val:
            extra = f"  intent={val['rendering_intent']}"
        elif key == "tEXt" and "text" in val:
            extra = f"  {val['text']!r}"
        elif key == "tIME" and "timestamp" in val:
            extra = f"  {val['timestamp']}"
        print(f"    {key:6s}: yes size={val['size']}{extra}")

    z = info["zlib"]
    hdr = z.get("header")
    print("  zlib:")
    if hdr is None:
        print(f"    header            : unavailable ({z.get('error', 'no data')})")
    else:
        print(
            f"    CMF/FLG           : {hdr['cmf']:02X}/{hdr['flg']:02X}  "
            f"CM={hdr['cm']} CINFO={hdr['cinfo']}  "
            f"window_bits={hdr['window_bits']} window={hdr['window_size']}  "
            f"FLEVEL={hdr['flevel']}({hdr['flevel_name']}) FDICT={hdr['fdict']} "
            f"FCHECK_ok={hdr['fcheck_ok']}"
        )
    if z.get("error"):
        print(f"    inflate           : FAIL {z['error']}")
    else:
        print(
            f"    inflate 5 rows    : "
            f"{'OK' if z.get('enough_for_rows') else 'PARTIAL'}  "
            f"got {z['inflated_bytes']}/{z['needed_bytes']} bytes  "
            f"fed {z['compressed_fed']} compressed from {z['chunks_used']} IDAT"
        )
    names = [FILTER_NAME.get(ft, f"?{ft}") for ft in info["filters"]]
    print(f"  first 5 filter bytes: {info['filters']}  ({names})")


def cell(v: object) -> str:
    return str(v)


def print_comparison(rows: list[tuple[str, dict | None]]) -> None:
    labels = [lab for lab, _ in rows]
    print()
    print("=" * 78)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 78)

    def get(info: dict | None, fn):
        if info is None:
            return "-"
        try:
            return fn(info)
        except Exception:
            return "-"

    fields = [
        ("file size", lambda i: fmt_bytes(i["file_size"])),
        ("width x height", lambda i: f"{i['ihdr']['width']} x {i['ihdr']['height']}"),
        ("bit_depth", lambda i: i["ihdr"]["bit_depth"]),
        ("color_type", lambda i: f"{i['ihdr']['color_type']} {i['ihdr']['color_type_name']}"),
        ("interlace", lambda i: i["ihdr"]["interlace"]),
        ("IHDR filter method", lambda i: i["ihdr"]["filter"]),
        ("row bytes (uncomp.)", lambda i: i["row_bytes"]),
        ("IDAT count", lambda i: i["idat_count"]),
        ("IDAT min", lambda i: i["idat_min"]),
        ("IDAT max", lambda i: i["idat_max"]),
        ("IDAT mean", lambda i: f"{i['idat_mean']:.1f}" if i["idat_mean"] is not None else "-"),
        ("IDAT total (comp.)", lambda i: fmt_bytes(i["idat_total"])),
        ("zlib window bits", lambda i: (i["zlib"].get("header") or {}).get("window_bits", "-")),
        ("zlib window size", lambda i: (i["zlib"].get("header") or {}).get("window_size", "-")),
        ("zlib FLEVEL", lambda i: (i["zlib"].get("header") or {}).get("flevel_name", "-")),
        ("first 5 filters", lambda i: i["filters"]),
        ("pHYs", lambda i: "yes" if i["ancillary"].get("pHYs") else "no"),
        ("gAMA", lambda i: "yes" if i["ancillary"].get("gAMA") else "no"),
        ("sRGB", lambda i: "yes" if i["ancillary"].get("sRGB") else "no"),
        ("iCCP", lambda i: "yes" if i["ancillary"].get("iCCP") else "no"),
        ("tEXt", lambda i: "yes" if i["ancillary"].get("tEXt") else "no"),
        ("tIME", lambda i: "yes" if i["ancillary"].get("tIME") else "no"),
        ("first 8 chunks", lambda i: i["first8"]),
        ("last 4 chunks", lambda i: i["last4"]),
    ]

    colw = [22] + [max(28, len(lab) + 2) for lab in labels]
    header = f"{'field':<22}" + "".join(f"{lab:<{w}}" for lab, w in zip(labels, colw[1:]))
    print(header)
    print("-" * len(header))
    for name, fn in fields:
        vals = [cell(get(info, fn)) for _, info in rows]
        line = f"{name:<22}" + "".join(f"{v:<{w}}" for v, w in zip(vals, colw[1:]))
        print(line)

    print()
    print("WHY A MAY DISPLAY FASTER / MORE COMPATIBLY THAN B")
    print("-" * 78)
    a = next((info for lab, info in rows if lab.startswith("A)")), None)
    b = next((info for lab, info in rows if lab.startswith("B)")), None)
    if a is None or b is None:
        print("A or B missing; cannot compare.")
        return

    notes: list[str] = []

    if a["idat_count"] != b["idat_count"] or (
        a["idat_max"] and b["idat_max"] and a["idat_max"] != b["idat_max"]
    ):
        a_max = a["idat_max"] or 0
        b_max = b["idat_max"] or 0
        notes.append(
            f"- IDAT layout: A has {a['idat_count']} chunks "
            f"(max {a_max} B, typical libpng ~8KiB) vs B has {b['idat_count']} "
            f"(max {b_max} B). Viewers that paint after each IDAT start sooner "
            "on ~8KiB chunks. ~1MiB IDATs delay first paint and some viewers "
            "only show the first chunk."
        )
    if a["idat_total"] and b["idat_total"]:
        ratio = b["idat_total"] / a["idat_total"] if a["idat_total"] else 0
        notes.append(
            f"- Compressed payload: A={fmt_bytes(a['idat_total'])} vs "
            f"B={fmt_bytes(b['idat_total'])} (B/A={ratio:.2f}x). "
            "Smaller zlib payload means less disk I/O and less inflate work."
        )

    a_f = a["filters"]
    b_f = b["filters"]
    a_none = a_f and all(x == 0 for x in a_f)
    b_none = b_f and all(x == 0 for x in b_f)
    a_adapt = a_f and any(x != 0 for x in a_f)
    b_adapt = b_f and any(x != 0 for x in b_f)
    if a_none and b_adapt:
        notes.append(
            "- Scanline filter: A uses filter 0 (None) on the first 5 rows; B uses "
            "adaptive filters. Filter 0 is cheaper to reconstruct (memcpy). Adaptive "
            "filters (Sub/Up/Average/Paeth) cost extra per-pixel math. Filter 0 can "
            "also compress worse, so this is a decode-speed vs file-size tradeoff."
        )
    elif b_none and a_adapt:
        notes.append(
            "- Scanline filter: B uses filter 0 (None) on the first 5 rows; A uses "
            "adaptive filters. Adaptive filtering usually shrinks IDAT (faster I/O) "
            "at the cost of slightly more CPU when undoing filters."
        )
    elif a_f != b_f:
        notes.append(f"- Scanline filters differ: A={a_f} B={b_f}.")

    if a["ihdr"]["interlace"] != b["ihdr"]["interlace"]:
        notes.append(
            f"- Interlace: A={a['ihdr']['interlace']} B={b['ihdr']['interlace']}. "
            "Adam7 (1) needs 7 passes and more seeking; non-interlaced (0) is "
            "simpler and usually faster to finish a full decode."
        )

    a_anc = {k for k in ANCILLARY_KEYS if a["ancillary"].get(k)}
    b_anc = {k for k in ANCILLARY_KEYS if b["ancillary"].get(k)}
    only_a = a_anc - b_anc
    only_b = b_anc - a_anc
    if only_a or only_b:
        notes.append(
            f"- Ancillary chunks: only in A={sorted(only_a) or '-'}  "
            f"only in B={sorted(only_b) or '-'}. "
            "Missing pHYs/gAMA/sRGB does not block decode, but viewers may assume "
            "wrong DPI or gamma. Extra iCCP can slow color-managed viewers."
        )

    ah = (a["zlib"].get("header") or {})
    bh = (b["zlib"].get("header") or {})
    if ah and bh:
        if ah.get("window_bits") != bh.get("window_bits"):
            notes.append(
                f"- zlib window: A={ah.get('window_bits')} bits "
                f"({ah.get('window_size')} B) vs B={bh.get('window_bits')} bits "
                f"({bh.get('window_size')} B). A smaller window uses less decoder "
                "memory; 15-bit (32KiB) is the PNG default and most compatible."
            )
        if ah.get("flevel") != bh.get("flevel"):
            notes.append(
                f"- zlib FLEVEL: A={ah.get('flevel_name')} vs B={bh.get('flevel_name')}. "
                "This is a compressor hint only; it does not change inflate algorithm."
            )

    if a["ihdr"]["width"] != b["ihdr"]["width"] or a["ihdr"]["height"] != b["ihdr"]["height"]:
        notes.append(
            f"- Dimensions differ: A={a['ihdr']['width']}x{a['ihdr']['height']} "
            f"B={b['ihdr']['width']}x{b['ihdr']['height']}. Larger images decode slower "
            "regardless of chunk layout."
        )

    if a["file_size"] < b["file_size"]:
        notes.append(
            f"- File size: A is smaller ({fmt_bytes(a['file_size'])} vs "
            f"{fmt_bytes(b['file_size'])}), so it maps/reads faster."
        )
    elif b["file_size"] < a["file_size"]:
        notes.append(
            f"- File size: B is smaller ({fmt_bytes(b['file_size'])} vs "
            f"{fmt_bytes(a['file_size'])}). If A still opens faster, the cause is "
            "more likely IDAT layout / filter / ancillary chunks than raw bytes."
        )

    for n in notes:
        print(n)


def main() -> int:
    import io
    from contextlib import redirect_stdout

    root = Path(__file__).resolve().parents[1]
    targets = [
        ("A) original", root / "original_colormap" / "phi250tenkaizu.png"),
        ("B) mode_A final", root / "data" / "output" / "three_direction" / "mode_A" / "final.png"),
        ("C) baseline", root / "original_colormap" / "phi250tenkaizu_baseline.png"),
    ]

    buf = io.StringIO()
    real_out = sys.stdout

    class _Tee:
        encoding = getattr(real_out, "encoding", "utf-8")

        def write(self, s: str) -> int:
            real_out.write(s)
            buf.write(s)
            return len(s)

        def flush(self) -> None:
            real_out.flush()

    results: list[tuple[str, dict | None]] = []
    sys.stdout = _Tee()  # type: ignore[assignment]
    try:
        for label, path in targets:
            if not path.is_file():
                print_report(label, None, path)
                results.append((label, None))
                continue
            info = analyze(path)
            print_report(label, info)
            results.append((label, info))
        print_comparison(results)
    finally:
        sys.stdout = real_out

    report_path = root / "scripts" / "png_structure_compare_out.txt"
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
