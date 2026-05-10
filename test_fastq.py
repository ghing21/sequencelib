"""
Tests for FASTQ support in sequencelib.
Run with: python3 test_fastq.py
"""

import gzip
import os
import sys
import tempfile
import traceback

sys.path.insert(0, "/home/claude")
import sequencelib as sl

PASSED = []
FAILED = []

def test(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()

# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_FASTQ = """\
@read1 length=10
ACGTACGTAC
+
IIIIIIIIII
@read2 some comment here
TTTTAAAACCGG
+read2
;;;;IIII;;;;
@read3
NNNNNNNN
+
########
"""

def make_plain_fastq(content=SAMPLE_FASTQ):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fastq", delete=False)
    f.write(content)
    f.close()
    return f.name

def make_gzip_fastq(content=SAMPLE_FASTQ):
    f = tempfile.NamedTemporaryFile(suffix=".fastq.gz", delete=False)
    f.close()
    with gzip.open(f.name, "wt") as gz:
        gz.write(content)
    return f.name

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fastq_sequence_class():
    seq = sl.Fastq_sequence("r1", "ACGT", "IIII")
    assert seq.name == "r1"
    assert seq.seq == "ACGT"
    assert seq.qual_string == "IIII"
    assert seq.qual_scores == [40, 40, 40, 40]

def test_fastq_sequence_mean_qual():
    seq = sl.Fastq_sequence("r1", "ACGT", "!III")  # 0,40,40,40
    assert seq.mean_qual() == 30.0

def test_fastq_sequence_min_qual():
    seq = sl.Fastq_sequence("r1", "ACGT", "!III")
    assert seq.min_qual() == 0

def test_fastq_sequence_length_mismatch():
    try:
        sl.Fastq_sequence("r1", "ACG", "IIII")
        assert False, "Should have raised SeqError"
    except sl.SeqError:
        pass

def test_fastq_sequence_fastq_repr():
    seq = sl.Fastq_sequence("r1", "ACGT", "IIII", comments="mycomment")
    fq = seq.fastq()
    assert fq.startswith("@r1 mycomment\n")
    assert "ACGT\n" in fq
    assert "+\n" in fq
    assert "IIII\n" in fq

def test_fastq_sequence_fasta_repr():
    seq = sl.Fastq_sequence("r1", "ACGT", "IIII")
    fa = seq.fasta()
    assert fa.startswith(">r1")
    assert "ACGT" in fa

def test_fastq_sequence_inherits_dna():
    seq = sl.Fastq_sequence("r1", "ACGT", "IIII")
    assert isinstance(seq, sl.DNA_sequence)

def test_fastqfilehandle_plain():
    fname = make_plain_fastq()
    try:
        reads = list(sl.Fastqfilehandle(fname))
        assert len(reads) == 3
        assert reads[0].name == "read1"
        assert reads[0].seq == "ACGTACGTAC"
        assert reads[0].qual_string == "IIIIIIIIII"
        assert reads[1].name == "read2"
        assert reads[1].comments == "some comment here"
        assert reads[2].name == "read3"
    finally:
        os.unlink(fname)

def test_fastqfilehandle_gzip():
    fname = make_gzip_fastq()
    try:
        reads = list(sl.Fastqfilehandle(fname))
        assert len(reads) == 3
        assert reads[0].name == "read1"
        assert reads[0].seq == "ACGTACGTAC"
    finally:
        os.unlink(fname)

def test_fastqfilehandle_is_generator():
    """Reads should be yielded one at a time (memory-efficient)."""
    fname = make_plain_fastq()
    try:
        fh = sl.Fastqfilehandle(fname)
        it = iter(fh)
        r1 = next(it)
        assert r1.name == "read1"
        r2 = next(it)
        assert r2.name == "read2"
    finally:
        os.unlink(fname)

def test_fastqfilehandle_read_seqs():
    fname = make_plain_fastq()
    try:
        seqset = sl.Fastqfilehandle(fname).read_seqs()
        assert len(seqset) == 3
    finally:
        os.unlink(fname)

def test_seqfile_factory_explicit():
    fname = make_plain_fastq()
    try:
        fh = sl.Seqfile(fname, filetype="fastq")
        reads = list(fh)
        assert len(reads) == 3
    finally:
        os.unlink(fname)

def test_seqfile_factory_autodetect_plain():
    fname = make_plain_fastq()
    try:
        fh = sl.Seqfile(fname)   # autodetect
        reads = list(fh)
        assert len(reads) == 3
        assert reads[0].name == "read1"
    finally:
        os.unlink(fname)

def test_seqfile_factory_autodetect_gzip():
    fname = make_gzip_fastq()
    try:
        fh = sl.Seqfile(fname)   # autodetect
        reads = list(fh)
        assert len(reads) == 3
    finally:
        os.unlink(fname)

def test_qual_scores_phred33():
    # '!' = ASCII 33 → Phred 0; 'I' = ASCII 73 → Phred 40
    seq = sl.Fastq_sequence("r", "AC", "!I")
    assert seq.qual_scores[0] == 0
    assert seq.qual_scores[1] == 40

def test_bad_fastq_no_at():
    fname = make_plain_fastq(">not_fastq\nACGT\n")
    try:
        try:
            list(sl.Fastqfilehandle(fname))
            assert False, "Expected SeqError"
        except sl.SeqError:
            pass
    finally:
        os.unlink(fname)

def test_seqfile_fasta_still_works():
    """Existing FASTA support should be unaffected."""
    content = ">seq1\nACGT\n>seq2\nTTTT\n"
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    f.write(content)
    f.close()
    try:
        seqs = list(sl.Seqfile(f.name))
        assert len(seqs) == 2
        assert seqs[0].name == "seq1"
    finally:
        os.unlink(f.name)

# ── Run all ────────────────────────────────────────────────────────────────────

print("\n=== FASTQ sequencelib test suite ===\n")
for name, fn in [
    ("Fastq_sequence: basic construction",       test_fastq_sequence_class),
    ("Fastq_sequence: mean_qual",                test_fastq_sequence_mean_qual),
    ("Fastq_sequence: min_qual",                 test_fastq_sequence_min_qual),
    ("Fastq_sequence: length mismatch raises",   test_fastq_sequence_length_mismatch),
    ("Fastq_sequence: fastq() repr",             test_fastq_sequence_fastq_repr),
    ("Fastq_sequence: fasta() repr",             test_fastq_sequence_fasta_repr),
    ("Fastq_sequence: inherits DNA_sequence",    test_fastq_sequence_inherits_dna),
    ("Fastqfilehandle: plain file",              test_fastqfilehandle_plain),
    ("Fastqfilehandle: gzip file",               test_fastqfilehandle_gzip),
    ("Fastqfilehandle: lazy generator",          test_fastqfilehandle_is_generator),
    ("Fastqfilehandle: read_seqs()",             test_fastqfilehandle_read_seqs),
    ("Seqfile: explicit filetype='fastq'",       test_seqfile_factory_explicit),
    ("Seqfile: autodetect plain .fastq",         test_seqfile_factory_autodetect_plain),
    ("Seqfile: autodetect gzip .fastq.gz",       test_seqfile_factory_autodetect_gzip),
    ("Phred+33 quality decoding",                test_qual_scores_phred33),
    ("Bad FASTQ raises SeqError",                test_bad_fastq_no_at),
    ("Existing FASTA support unaffected",        test_seqfile_fasta_still_works),
]:
    test(name, fn)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    sys.exit(1)