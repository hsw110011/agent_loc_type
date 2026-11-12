"""缓存命名：新名字带时间和图名，按哈希能找回，旧的纯哈希命名继续命中。"""
import re
import tempfile
import unittest
from pathlib import Path

from agent.map_preprocess import output_layout as layout

KEY = 'ab'*32


class CacheNameTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_new_name_has_time_and_image(self):
        name = layout.cache_name('data/kitti_raw_1003_0027.tif', KEY, '.npz')
        self.assertRegex(name, r'^\d{8}_\d{6}_kitti_raw_1003_0027_abababababab\.npz$')

    def test_named_entries_are_found_by_key(self):
        target = self.root/layout.cache_name('a.tif', KEY)
        target.mkdir()
        self.assertEqual(layout.find_cache(self.root, KEY), target)
        self.assertIsNone(layout.find_cache(self.root, 'cd'*32))

    def test_latest_of_several_matches_wins(self):
        for stamp in ('20260101_000000', '20261231_000000'):
            (self.root/f'{stamp}_a_{KEY[:12]}.npz').touch()
        self.assertTrue(re.match('20261231', layout.find_cache(self.root, KEY, '.npz').name))

    def test_legacy_hash_names_still_hit(self):
        (self.root/KEY).mkdir()
        self.assertEqual(layout.find_cache(self.root, KEY), self.root/KEY)
        (self.root/f'{KEY[:24]}.npz').touch()
        self.assertEqual(layout.find_cache(self.root, KEY, '.npz'), self.root/f'{KEY[:24]}.npz')

    def test_missing_root_is_not_an_error(self):
        self.assertIsNone(layout.find_cache(self.root/'nope', KEY))


if __name__ == '__main__':
    unittest.main()
