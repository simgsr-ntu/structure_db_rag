import pytest
from src.ingestion.sermon_grouper import group_sermon_files


class TestGroupSermonFiles:
    def test_pairs_ng_with_matching_ps_by_date(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert files[1] in groups[0].ps

    def test_pairs_by_topic_when_slide_has_date_proximity(self):
        files = [
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert files[1] in groups[0].ps

    def test_standalone_ps_without_ng(self):
        files = ["English_2018_20180623-Growing-Faith-in-God-Final-PPT.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng is None
        assert files[0] in groups[0].ps

    def test_standalone_ng_without_ps(self):
        files = ["English_2018_28-29-Jul-2018-Know-Your-Enemy-by-Elder-Edric-Sng-Members-guide-updated.pdf"]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert groups[0].ng == files[0]
        assert groups[0].ps == []

    def test_does_not_pair_different_weekends(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf",
            "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf",
            "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 2
        ng_ps = {g.ng: g.ps for g in groups}
        assert "English_2018_FinishingWell_DSP_2018-06-02_03_r1.pdf" in \
               ng_ps["English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf"]
        assert "English_2018_An-Altar-Not-To-Miss-9-June-2018.pdf" in \
               ng_ps["English_2018_09-10-June-2018-An-Altar-Not-to-Miss-by-Ps-Jason-Teo-Members-Guide.pdf"]

    def test_handouts_are_ignored(self):
        files = [
            "English_2018_02-03-June-2018-Finishing-Well-by-DSP-Members-Guide.pdf",
            "English_2018_FinishingWell_Handout.pdf",
        ]
        groups = group_sermon_files(files)
        assert len(groups) == 1
        assert "English_2018_FinishingWell_Handout.pdf" not in groups[0].ps
