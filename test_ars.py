import ars

oneLineFixtureArsAway = '<div class="fixture__teams">Leeds v Arsenal </div>'
oneLineFixtureArsHome = '<div class="fixture__teams">Arsenal v Leeds </div>'
oneLineChannel = 'style="background-color: #e31a00;border: 0;color: rgba(255, 255, 255, 1.0);">Sky Sports Main Event</span><span class="channel-pill'
oneLineDate = '<div class="fixture-date">Friday 5th August 2022</div>'
oneLineTime = '<div class="fixture__time">20:00</div>'


def test_should_return_fixture_when_at_home():
    assert ars.returnNextFixture(oneLineFixtureArsAway) == "Leeds"


def test_should_return_first_fixture_when_away():
    assert ars.returnNextFixture(oneLineFixtureArsHome) == "Leeds"


def test_should_return_first_fixture_when_more_than_one_and_at_home():
    testData = open('testData/arsHome.txt')
    multipleFixtures = testData.read()
    testData.close()

    assert ars.returnNextFixture(multipleFixtures) == "Leeds"


def test_should_return_first_fixture_when_more_than_one_and_away():
    testData = open('testData/arsAway.txt')
    multipleFixtures = testData.read()
    testData.close()

    assert ars.returnNextFixture(multipleFixtures) == "Leeds"


def test_should_return_channel():
    assert ars.returnChannel(oneLineChannel) == "Sky Sports Main Event"


def test_should_return_channel_when_more_than_one():
    testData = open('testData/arsChannel.txt')
    multipleFixtures = testData.read()
    testData.close()

    assert ars.returnChannel(multipleFixtures) == "Sky Sports Main Event"


def test_should_return_date():
    assert ars.returnDate(oneLineDate) == "Friday 5th August 2022"


def test_should_return_first_date_when_more_than_one():
    testData = open('testData/arsHome.txt')
    multipleFixtures = testData.read()
    testData.close()

    assert ars.returnDate(multipleFixtures) == "Friday 5th August 2022"


def test_should_return_time():
    assert ars.returnTime(oneLineTime) == "20:00"


def test_should_return_first_date_when_more_than_one():
    testData = open('testData/arsHome.txt')
    multipleFixtures = testData.read()
    testData.close()

    assert ars.returnTime(multipleFixtures) == "20:00"
