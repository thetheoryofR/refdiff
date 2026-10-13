from refdiff import data


def test_read_boxes_spaces_and_commas(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("466 441 493 470 3\n10,20,5,8,6\n\n")
    b = data.read_boxes(p)
    assert b.tolist() == [[466, 441, 493, 470, 2], [5, 8, 10, 20, 5]]


def test_split_parses_list_without_trailing_newline(tmp_path):
    (tmp_path / "test.txt").write_text("group1/1/11.jpg group1/1_not/11.txt\ngroup1/1/12.jpg group1/1_not/12.txt")
    pairs = data.split("test", tmp_path)
    assert [p.id for p in pairs] == ["11", "12"]
    assert pairs[0].test.name == "11_test.jpg" and pairs[0].temp.name == "11_temp.jpg"
