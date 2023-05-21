import unittest

import zstandard as zstd


class TestDecompressor_decompress(unittest.TestCase):
    def test_empty_input(self):
        dctx = zstd.ZstdDecompressor()

        with self.assertRaisesRegex(
            zstd.ZstdError, "error determining content size from frame header"
        ):
            dctx.decompress(b"")

    def test_invalid_input(self):
        dctx = zstd.ZstdDecompressor()

        with self.assertRaisesRegex(
            zstd.ZstdError, "error determining content size from frame header"
        ):
            dctx.decompress(b"foobar")

    def test_input_types(self):
        cctx = zstd.ZstdCompressor(level=1)
        compressed = cctx.compress(b"foo")

        mutable_array = bytearray(len(compressed))
        mutable_array[:] = compressed

        sources = [
            memoryview(compressed),
            bytearray(compressed),
            mutable_array,
        ]

        dctx = zstd.ZstdDecompressor()
        for source in sources:
            self.assertEqual(dctx.decompress(source), b"foo")

    def test_no_content_size_in_frame(self):
        cctx = zstd.ZstdCompressor(write_content_size=False)
        compressed = cctx.compress(b"foobar")

        dctx = zstd.ZstdDecompressor()
        with self.assertRaisesRegex(
            zstd.ZstdError, "could not determine content size in frame header"
        ):
            dctx.decompress(compressed)

    def test_content_size_present(self):
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(b"foobar")

        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        self.assertEqual(decompressed, b"foobar")

    def test_empty_roundtrip(self):
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(b"")

        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)

        self.assertEqual(decompressed, b"")

    def test_max_output_size(self):
        cctx = zstd.ZstdCompressor(write_content_size=False)
        source = b"foobar" * 256
        compressed = cctx.compress(source)

        dctx = zstd.ZstdDecompressor()
        # Will fit into buffer exactly the size of input.
        decompressed = dctx.decompress(compressed, max_output_size=len(source))
        self.assertEqual(decompressed, source)

        # Input size - 1 fails
        with self.assertRaisesRegex(
            zstd.ZstdError, "decompression error: did not decompress full frame"
        ):
            dctx.decompress(compressed, max_output_size=len(source) - 1)

        # Input size + 1 works
        decompressed = dctx.decompress(
            compressed, max_output_size=len(source) + 1
        )
        self.assertEqual(decompressed, source)

        # A much larger buffer works.
        decompressed = dctx.decompress(
            compressed, max_output_size=len(source) * 64
        )
        self.assertEqual(decompressed, source)

    def test_stupidly_large_output_buffer(self):
        cctx = zstd.ZstdCompressor(write_content_size=False)
        compressed = cctx.compress(b"foobar" * 256)
        dctx = zstd.ZstdDecompressor()

        # Will get OverflowError on some Python distributions that can't
        # handle really large integers.
        with self.assertRaises((MemoryError, OverflowError)):
            dctx.decompress(compressed, max_output_size=2**62)

    def test_dictionary(self):
        samples = []
        for i in range(128):
            samples.append(b"foo" * 64)
            samples.append(b"bar" * 64)
            samples.append(b"foobar" * 64)
            samples.append(b"qwert" * 64)
            samples.append(b"yuiop" * 64)
            samples.append(b"asdfg" * 64)
            samples.append(b"hijkl" * 64)

        d = zstd.train_dictionary(8192, samples)

        orig = b"foobar" * 16384
        cctx = zstd.ZstdCompressor(level=1, dict_data=d)
        compressed = cctx.compress(orig)

        dctx = zstd.ZstdDecompressor(dict_data=d)
        decompressed = dctx.decompress(compressed)

        self.assertEqual(decompressed, orig)

    def test_dictionary_multiple(self):
        samples = []
        for i in range(128):
            samples.append(b"foo" * 64)
            samples.append(b"bar" * 64)
            samples.append(b"foobar" * 64)
            samples.append(b"qwert" * 64)
            samples.append(b"yuiop" * 64)
            samples.append(b"asdfg" * 64)
            samples.append(b"hijkl" * 64)

        d = zstd.train_dictionary(8192, samples)

        sources = (b"foobar" * 8192, b"foo" * 8192, b"bar" * 8192)
        compressed = []
        cctx = zstd.ZstdCompressor(level=1, dict_data=d)
        for source in sources:
            compressed.append(cctx.compress(source))

        dctx = zstd.ZstdDecompressor(dict_data=d)
        for i in range(len(sources)):
            decompressed = dctx.decompress(compressed[i])
            self.assertEqual(decompressed, sources[i])

    def test_max_window_size(self):
        with open(__file__, "rb") as fh:
            source = fh.read()

        # If we write a content size, the decompressor engages single pass
        # mode and the window size doesn't come into play.
        cctx = zstd.ZstdCompressor(write_content_size=False)
        frame = cctx.compress(source)

        dctx = zstd.ZstdDecompressor(max_window_size=2**zstd.WINDOWLOG_MIN)

        with self.assertRaisesRegex(
            zstd.ZstdError,
            "decompression error: Frame requires too much memory",
        ):
            dctx.decompress(frame, max_output_size=len(source))

    def test_explicit_default_params(self):
        cctx = zstd.ZstdCompressor(level=1)
        compressed = cctx.compress(b"foo")

        dctx = zstd.ZstdDecompressor(
            dict_data=None,
            max_window_size=0,
            format=zstd.FORMAT_ZSTD1,
        )
        self.assertEqual(dctx.decompress(compressed), b"foo")

    def test_read_across_frames_and_allow_extra_data(self):
        with self.assertRaisesRegex(
            zstd.ZstdError,
            "read_across_frames and allow_extra_data cannot both be true",
        ):
            zstd.ZstdDecompressor().decompress(
                b"irrelevant", read_across_frames=True, allow_extra_data=True
            )

    def test_multiple_frames(self):
        cctx = zstd.ZstdCompressor()
        foo = cctx.compress(b"foo")
        bar = cctx.compress(b"bar")

        dctx = zstd.ZstdDecompressor()
        self.assertEqual(dctx.decompress(foo + bar), b"foo")
        self.assertEqual(
            dctx.decompress(foo + bar, allow_extra_data=True), b"foo"
        )
        self.assertEqual(
            dctx.decompress(
                foo + bar, read_across_frames=True, allow_extra_data=False
            ),
            b"foobar",
        )

    def test_multiple_frames_max_output_size(self):
        """A subsequent frame overflowing the max output size raises an error."""
        cctx = zstd.ZstdCompressor()

        foo_source = b"foo" * 1024
        bar_source = b"bar" * 128

        foo_frame = cctx.compress(foo_source)
        bar_frame = cctx.compress(bar_source)

        total_input_size = len(foo_source) + len(bar_source)

        dctx = zstd.ZstdDecompressor()

        with self.assertRaisesRegex(
            zstd.ZstdError,
            "max allowed output size reached; would read up to 3456 bytes",
        ):
            dctx.decompress(
                foo_frame + bar_frame,
                read_across_frames=True,
                allow_extra_data=False,
                max_output_size=len(foo_source) + 8,
            )

    def test_junk_after_frame(self):
        cctx = zstd.ZstdCompressor()
        frame = cctx.compress(b"foo")

        dctx = zstd.ZstdDecompressor()
        self.assertEqual(dctx.decompress(frame + b"junk"), b"foo")

        self.assertEqual(
            dctx.decompress(frame + b"junk", allow_extra_data=True), b"foo"
        )

        with self.assertRaisesRegex(
            zstd.ZstdError, "4 bytes of unused data, which is disallowed"
        ):
            dctx.decompress(frame + b"junk", allow_extra_data=False)

        with self.assertRaisesRegex(
            zstd.ZstdError,
            "error determining content size from frame header",
        ):
            dctx.decompress(
                frame + b"junk", read_across_frames=True, allow_extra_data=False
            )

    def test_data_after_empty_frame(self):
        cctx = zstd.ZstdCompressor()

        empty_frame = cctx.compress(b"")
        foo_frame = cctx.compress(b"foo")

        dctx = zstd.ZstdDecompressor()

        self.assertEqual(
            dctx.decompress(empty_frame + foo_frame, allow_extra_data=True),
            b"",
        )

        with self.assertRaisesRegex(
            zstd.ZstdError,
            "compressed input contains 12 bytes of unused data, which is disallowed",
        ):
            dctx.decompress(empty_frame + foo_frame, allow_extra_data=False)

        self.assertEqual(
            dctx.decompress(
                empty_frame + foo_frame,
                read_across_frames=True,
                allow_extra_data=False,
            ),
            b"foo",
        )
